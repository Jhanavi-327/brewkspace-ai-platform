import os
import cv2
import numpy as np
from typing import Optional


STANDARD_WIDTH = 1000
STANDARD_HEIGHT = 630


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


def four_point_transform(image, points):
    rect = order_points(points)

    dst = np.array(
        [
            [0, 0],
            [STANDARD_WIDTH - 1, 0],
            [STANDARD_WIDTH - 1, STANDARD_HEIGHT - 1],
            [0, STANDARD_HEIGHT - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)

    return cv2.warpPerspective(
        image,
        matrix,
        (STANDARD_WIDTH, STANDARD_HEIGHT)
    )


def find_card_contour(image) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edged = cv2.Canny(
        blurred,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    image_area = image.shape[0] * image.shape[1]

    for contour in contours[:10]:
        area = cv2.contourArea(contour)

        if area < image_area * 0.15:
            continue

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")

    return None


def center_crop_fallback(image):
    h, w = image.shape[:2]

    target_ratio = STANDARD_WIDTH / STANDARD_HEIGHT
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        cropped = image[:, x1:x1 + new_w]
    else:
        new_h = int(w / target_ratio)
        y1 = (h - new_h) // 2
        cropped = image[y1:y1 + new_h, :]

    return cv2.resize(
        cropped,
        (STANDARD_WIDTH, STANDARD_HEIGHT),
        interpolation=cv2.INTER_CUBIC
    )


def normalize_aadhaar_card_image(
    image_path: str,
    debug_dir: Optional[str] = None
) -> str:
    from app.services.kyc.image_loader import load_document_as_bgr

    image = load_document_as_bgr(image_path)

    if image is None:
        raise ValueError(
            f"Could not read Aadhaar image: {image_path}"
        )

    contour = find_card_contour(image)

    if contour is not None:
        normalized = four_point_transform(
            image,
            contour
        )
        method = "perspective"
    else:
        normalized = center_crop_fallback(image)
        method = "center_crop"

    if debug_dir is None:
        debug_dir = os.path.join(
            os.path.dirname(image_path),
            "debug_aadhaar"
        )

    os.makedirs(debug_dir, exist_ok=True)

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = os.path.join(
        debug_dir,
        f"{base_name}_normalized_aadhaar_{method}.jpg"
    )

    cv2.imwrite(output_path, normalized)

    print(
        "AADHAAR CARD NORMALIZED:",
        output_path,
        "METHOD:",
        method
    )
    

    return output_path