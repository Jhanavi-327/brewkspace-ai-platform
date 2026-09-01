import cv2
import pytesseract
from PIL import Image


def preprocess_pan_image(image_path: str):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read PAN image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return thresh


def run_pan_ocr(image_path: str) -> str:
    processed = preprocess_pan_image(image_path)

    text = pytesseract.image_to_string(
        processed,
        config="--psm 6"
    )

    return text