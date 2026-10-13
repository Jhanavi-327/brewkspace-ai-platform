import cv2
import pytesseract
import easyocr
import numpy as np


_reader = None


def get_easyocr_reader():
    global _reader

    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)

    return _reader


def resize_image(image, scale=2.5):
    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


def preprocess_variants(image):
    variants = {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    variants["gray"] = resize_image(gray)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    variants["clahe"] = resize_image(clahe.apply(gray))

    otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    variants["otsu"] = resize_image(otsu)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )
    variants["adaptive"] = resize_image(adaptive)

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharp = cv2.filter2D(gray, -1, kernel)
    variants["sharp"] = resize_image(sharp)

    return variants


def run_tesseract(image) -> str:
    outputs = []

    for config in ["--psm 6", "--psm 11", "--psm 12"]:
        try:
            text = pytesseract.image_to_string(
                image,
                config=config
            )

            if text.strip():
                outputs.append(
                    f"\n--- TESSERACT {config} ---\n{text}"
                )

        except Exception as e:
            outputs.append(
                f"\n--- TESSERACT ERROR {config}: {e} ---"
            )

    return "\n".join(outputs)


def run_easyocr(image) -> str:
    try:
        reader = get_easyocr_reader()

        results = reader.readtext(
            image,
            detail=0,
            paragraph=False
        )

        return "\n".join(results)

    except Exception as e:
        return f"EASYOCR ERROR: {e}"


def run_aadhaar_ocr(image_path: str) -> str:

    from app.services.kyc.aadhaar_preprocessor import (
        normalize_aadhaar_card_image
    )

    from app.services.kyc.image_loader import (
        load_document_as_bgr
    )

    # Normalize the Aadhaar card first
    normalized_path = normalize_aadhaar_card_image(
        image_path
    )

    image = load_document_as_bgr(
        normalized_path
    )

    if image is None:
        raise ValueError(
            "Could not read normalized Aadhaar image"
        )

    outputs = []

    for name, variant in preprocess_variants(image).items():

        outputs.append(
            f"\n========== AADHAAR PREPROCESS VARIANT: {name} =========="
        )

        outputs.append(
            run_tesseract(variant)
        )

        outputs.append(
            f"\n--- EASYOCR {name} ---\n{run_easyocr(variant)}"
        )

    print(
        "AADHAAR MULTI-PREPROCESS OCR COMPLETE"
    )

    return "\n".join(outputs)