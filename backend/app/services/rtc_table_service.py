import cv2
import fitz
import numpy as np

from app.services.ocr_service import auto_rotate_for_ocr

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


def preprocess_image(image_np):
    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
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
        cv2.THRESH_BINARY_INV,
        31,
        15
    )

    return thresh


def detect_table_grid(thresh):
    image_width = thresh.shape[1]
    image_height = thresh.shape[0]

    horizontal_kernel_length = max(
        20,
        image_width // 90
    )

    vertical_kernel_length = max(
        20,
        image_height // 90
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_kernel_length, 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_kernel_length)
    )

    horizontal_lines = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1
    )

    vertical_lines = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1
    )

    table_grid = cv2.add(
        horizontal_lines,
        vertical_lines
    )

    table_grid = cv2.dilate(
        table_grid,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    return table_grid


def box_area(box: Dict[str, Any]) -> int:
    return box["width"] * box["height"]


def remove_duplicate_boxes(
    boxes: List[Dict[str, Any]],
    tolerance: int = 8
):
    unique = []

    for box in boxes:
        duplicate = False

        for existing in unique:
            if (
                abs(box["x"] - existing["x"]) < tolerance
                and abs(box["y"] - existing["y"]) < tolerance
                and abs(box["width"] - existing["width"]) < tolerance * 2
                and abs(box["height"] - existing["height"]) < tolerance * 2
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append(box)

    return unique


def remove_outer_container_boxes(
    boxes: List[Dict[str, Any]],
    image_width: int,
    image_height: int
) -> List[Dict[str, Any]]:

    cleaned = []
    page_area = image_width * image_height

    for box in boxes:
        area = box_area(box)

        if area > page_area * 0.20:
            continue

        if (
            box["width"] > image_width * 0.90
            and box["height"] > image_height * 0.40
        ):
            continue

        cleaned.append(box)

    return cleaned


def extract_cell_boxes(
    table_grid,
    image_width: int,
    image_height: int
):
    contours, _ = cv2.findContours(
        table_grid,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < 20 or h < 15:
            continue

        if w > image_width * 0.90:
            continue

        if h > image_height * 0.40:
            continue

        area = w * h

        if area > image_width * image_height * 0.15:
            continue

        boxes.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "x2": int(x + w),
            "y2": int(y + h)
        })

    boxes = remove_duplicate_boxes(boxes)

    boxes = remove_outer_container_boxes(
        boxes,
        image_width,
        image_height
    )

    boxes = sorted(
        boxes,
        key=lambda box: (box["y"], box["x"])
    )

    return boxes


def group_cells_into_rows(
    boxes,
    row_threshold: int = 18
):
    rows = []

    for box in boxes:
        placed = False

        for row in rows:
            row_y = row[0]["y"]

            if abs(row_y - box["y"]) <= row_threshold:
                row.append(box)
                placed = True
                break

        if not placed:
            rows.append([box])

    rows.sort(key=lambda row: row[0]["y"])

    for row_index, row in enumerate(rows):
        row.sort(key=lambda box: box["x"])

        for column_index, box in enumerate(row):
            box["row"] = row_index
            box["column"] = column_index

    return rows


def detect_rtc_regions(
    rows,
    image_height: int
):
    regions = {
        "top_header_cells": [],
        "main_rtc_table": [],
        "cultivation_table": []
    }

    for row in rows:
        avg_y = sum(
            cell["y"]
            for cell in row
        ) / len(row)

        if avg_y < image_height * 0.18:
            regions["top_header_cells"].append(row)

        elif avg_y < image_height * 0.50:
            regions["main_rtc_table"].append(row)

        else:
            regions["cultivation_table"].append(row)

    return regions


def save_debug_table_image(
    table_grid,
    cells: List[Dict[str, Any]],
    output_path: str = "rtc_table_debug.png"
):
    debug_image = cv2.cvtColor(
        table_grid,
        cv2.COLOR_GRAY2BGR
    )

    for cell in cells:
        cv2.rectangle(
            debug_image,
            (cell["x"], cell["y"]),
            (cell["x2"], cell["y2"]),
            (0, 255, 0),
            2
        )

    cv2.imwrite(
        output_path,
        debug_image
    )


def extract_rtc_table_layout(file_path: str) -> Dict[str, Any]:

    image_np = pdf_page_to_image(file_path)

    image_height = image_np.shape[0]
    image_width = image_np.shape[1]

    thresh = preprocess_image(image_np)

    table_grid = detect_table_grid(thresh)

    cells = extract_cell_boxes(
        table_grid,
        image_width,
        image_height
    )

    rows = group_cells_into_rows(cells)

    regions = detect_rtc_regions(
        rows,
        image_height
    )

    save_debug_table_image(
        table_grid,
        cells
    )

    print(
        f"RTC TABLE DETECTION: "
        f"cells={len(cells)} "
        f"rows={len(rows)}"
    )

    return {
        "page": 1,
        "image_width": image_width,
        "image_height": image_height,
        "cell_count": len(cells),
        "row_count": len(rows),
        "cells": cells,
        "rows": rows,
        "regions": regions
    }