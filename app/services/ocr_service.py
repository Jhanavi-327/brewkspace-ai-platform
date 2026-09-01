import io
import os
import re
import fitz
import cv2
import torch
import easyocr
import pytesseract
import numpy as np
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TESSERACT_LANG = "kan+eng"
EASYOCR_LANGS = ["kn", "en"]

# Render DPI for PDF pages. Also used to compare a rendered image's pixel width
# against the source image's native width, so the two must stay in step.
_RENDER_DPI = 350


easyocr_reader = easyocr.Reader(
    EASYOCR_LANGS,
    gpu=torch.cuda.is_available()
)


def _to_gray(image_np):
    """Grayscale only — no denoise, no threshold.

    preprocess_for_ocr is tuned for large scans: it denoises hard
    (fastNlMeansDenoising h=30) and adaptively thresholds over a 31px window.
    Applied to a SMALL image that has already been magnified, that erases the
    thin strokes it is meant to clean, turning readable text into noise. On the
    367x232 Aadhaar scan the same crop OCR'd to 361 useful characters plain and
    to garbage after preprocessing.
    """
    if len(image_np.shape) == 2:
        return image_np

    return cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)


def preprocess_for_ocr(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    threshold = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    return threshold


def run_tesseract(image_np, preprocess: bool = True) -> str:
    processed = preprocess_for_ocr(image_np) if preprocess else _to_gray(image_np)

    text = pytesseract.image_to_string(
        processed,
        lang=TESSERACT_LANG,
        config="--psm 6"
    )

    return text.strip()


def run_easyocr(image_np, preprocess: bool = True) -> str:
    processed = preprocess_for_ocr(image_np) if preprocess else _to_gray(image_np)

    results = easyocr_reader.readtext(
        processed,
        detail=0,
        paragraph=False,
        decoder="beamsearch",
        width_ths=0.7,
        text_threshold=0.4,
        low_text=0.3,
        link_threshold=0.3
    )

    return "\n".join(results).strip()


def readable_score(text: str) -> int:
    if not text:
        return 0

    kannada_chars = sum(1 for char in text if "\u0C80" <= char <= "\u0CFF")
    english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
    digits = sum(1 for char in text if char.isdigit())
    useful_symbols = text.count("/") + text.count("-") + text.count(".")

    return kannada_chars + english_chars + digits + useful_symbols


def contains_rtc_keywords(text: str) -> bool:
    if not text:
        return False

    keywords = [
        "ರೆಕಾರ್ಡ್",
        "ರೈಟ್ಸ್",
        "ಪಹಣಿ",
        "ತಾಲ್ಲೂಕು",
        "ಹೋಬಳಿ",
        "ಹೋಬಲಿ",
        "ಗ್ರಾಮ",
        "ಸರ್ವೆ",
        "ಖಾತೆ",
        "ಕಂದಾಯ",
        "MR",
        "R.T.C",
        "RTC",
        "Village Account Form",
        "landrecords.karnataka.gov",
        "PreviewRTC",
    ]

    text_lower = text.lower()

    return any(keyword.lower() in text_lower for keyword in keywords)


def run_hybrid_ocr(image_np, preprocess: bool = True) -> str:
    tesseract_text = run_tesseract(image_np, preprocess=preprocess)
    easyocr_text = run_easyocr(image_np, preprocess=preprocess)

    if contains_rtc_keywords(easyocr_text):
        selected = easyocr_text
        selected_engine = "EASYOCR"
    elif contains_rtc_keywords(tesseract_text):
        selected = tesseract_text
        selected_engine = "TESSERACT"
    else:
        tesseract_score = readable_score(tesseract_text)
        easyocr_score = readable_score(easyocr_text)

        if easyocr_score >= tesseract_score:
            selected = easyocr_text
            selected_engine = "EASYOCR"
        else:
            selected = tesseract_text
            selected_engine = "TESSERACT"

    return (
        f"\n--- SELECTED OCR ENGINE: {selected_engine} ---\n"
        f"{selected}\n\n"
        f"--- TESSERACT OCR ---\n"
        f"{tesseract_text}\n\n"
        f"--- EASYOCR OCR ---\n"
        f"{easyocr_text}"
    )


def pdf_page_to_image(page, dpi: int = _RENDER_DPI):
    pix = page.get_pixmap(dpi=dpi)

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    return np.array(image)


# A rendered page can be unreadable even at high DPI when the PDF places a
# SMALL embedded image on a large page: rendering then upscales those few
# original pixels, which adds blur rather than detail. Observed on an Aadhaar
# PDF whose two card scans are 367x232 and 370x235 dropped onto a 596x842 A4
# page — the 350 DPI page render OCR'd to a single character, while OCR'ing the
# embedded images at their native resolution recovered the name, DOB and
# gender. Below this many readable characters a page counts as failed and the
# embedded images are tried directly.
_PAGE_TEXT_FAILURE_SCORE = 40

# Small embedded scans need magnifying for Tesseract, which wants roughly
# 30px-tall glyphs. Larger ones are already adequate and upscaling them would
# only cost time.
_EMBEDDED_MIN_WIDTH = 1000
_EMBEDDED_UPSCALE = 4

# Magnification above which the render is mostly interpolation and the source
# image is worth OCR'ing directly. Measured across this project's documents:
# generated RTC PDFs sit at ~1.4x, photographed ID cards at 2.4x-5.7x.
_EMBEDDED_UPSCALE_TRIGGER = 2.0


def ocr_content_score(wrapped_text: str) -> int:
    """
    Score only the OCR CONTENT of a run_hybrid_ocr block.

    run_hybrid_ocr wraps its output in "--- SELECTED OCR ENGINE: X ---" /
    "--- TESSERACT OCR ---" / "--- EASYOCR OCR ---" markers. Those markers are
    themselves ~78 readable ASCII characters, so scoring the wrapped block
    directly gives an EMPTY page a healthy-looking score and any
    quality gate keyed on it never fires. Strip the markers first.
    """
    if not wrapped_text:
        return 0

    content = re.sub(r"-{2,}[^\n]*-{2,}", " ", wrapped_text)

    return readable_score(content)


def embedded_page_images(pdf_document, page):
    """Yield each image embedded in `page`, at its NATIVE resolution."""
    for image_info in page.get_images(full=True):
        xref = image_info[0]

        try:
            extracted = pdf_document.extract_image(xref)
            image = Image.open(io.BytesIO(extracted["image"])).convert("RGB")
        except Exception as error:
            print("Embedded image extraction failed:", error)
            continue

        yield np.array(image)


def page_image_upscale(pdf_document, page) -> float:
    """
    Largest factor by which rendering MAGNIFIES an embedded image on this page.

    A factor above 1 means the renderer is interpolating: the page is drawn at
    more pixels than the source image actually has, so those extra pixels carry
    no new detail, only blur. Reading the source image directly then costs one
    OCR pass and can only help. Below 1 the render already has at least the
    source's resolution and there is nothing to gain.
    """
    largest = 0.0

    for image_info in page.get_images(full=True):
        xref = image_info[0]

        try:
            native_width = pdf_document.extract_image(xref)["width"]
            rects = page.get_image_rects(xref)
        except Exception:
            continue

        if not native_width or not rects:
            continue

        rendered_width = max(rect.width for rect in rects) * _RENDER_DPI / 72.0

        largest = max(largest, rendered_width / native_width)

    return largest


def should_read_embedded_images(pdf_document, page, page_text: str) -> bool:
    """
    Whether this page's embedded images are worth a separate OCR pass.

    Two independent triggers:
      * the rendered page produced almost no text at all, or
      * the render is magnifying an embedded image beyond
        _EMBEDDED_UPSCALE_TRIGGER, so the rendered pixels are interpolation.

    The upscale test is what separates document classes here without naming
    them: generated RTC PDFs embed a 2800px scan and magnify it only ~1.4x, so
    they never take the extra pass, while photographed ID cards embed 367-1700px
    scans magnified 2.4x-5.7x and always do.
    """
    if ocr_content_score(page_text) < _PAGE_TEXT_FAILURE_SCORE:
        return True

    return page_image_upscale(pdf_document, page) >= _EMBEDDED_UPSCALE_TRIGGER


def ocr_embedded_images(pdf_document, page) -> str:
    """
    OCR a page's embedded images directly, for pages whose rendered form came
    out unreadable. Returns "" when the page embeds no images.
    """
    parts = []

    for image_np in embedded_page_images(pdf_document, page):
        width = image_np.shape[1]

        if width < _EMBEDDED_MIN_WIDTH:
            image_np = cv2.resize(
                image_np,
                None,
                fx=_EMBEDDED_UPSCALE,
                fy=_EMBEDDED_UPSCALE,
                interpolation=cv2.INTER_CUBIC,
            )

        parts.append(run_hybrid_ocr(image_np, preprocess=False))

    return "\n".join(parts)


def extract_text_from_pdf(file_path: str) -> str:
    text_parts = []

    pdf_document = fitz.open(file_path)

    for page_index in range(len(pdf_document)):
        page = pdf_document[page_index]
        image_np = pdf_page_to_image(page)

        page_text = run_hybrid_ocr(image_np)

        if should_read_embedded_images(pdf_document, page, page_text):
            embedded_text = ocr_embedded_images(pdf_document, page)

            if embedded_text.strip():
                # APPEND, never replace. The two reads fail differently — on a
                # scanned Aadhaar back, the render produced noise that scored
                # HIGHER than the embedded read (character counts cannot tell
                # noise from text) while only the embedded read contained the
                # address. Keeping both lets the field extractors take whichever
                # actually carries each value.
                page_text = (
                    f"{page_text}\n\n"
                    f"--- EMBEDDED IMAGE OCR ---\n{embedded_text}"
                )

        text_parts.append(f"\n--- PAGE {page_index + 1} OCR ---\n")
        text_parts.append(page_text)

    pdf_document.close()

    return "\n".join(text_parts)


def extract_text_from_image(file_path: str) -> str:
    image = Image.open(file_path).convert("RGB")
    image_np = np.array(image)

    return run_hybrid_ocr(image_np)


def extract_text(file_path: str, file_type: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type = file_type.upper()

    if file_type == "PDF":
        return extract_text_from_pdf(file_path)

    if file_type in ["JPEG", "JPG", "PNG", "TIFF", "TIF"]:
        return extract_text_from_image(file_path)

    raise ValueError(f"Unsupported file type for OCR: {file_type}")