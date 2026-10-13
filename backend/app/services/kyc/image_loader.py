import os
import cv2
import numpy as np
import fitz


def load_document_as_bgr(path: str, dpi: int = 300) -> np.ndarray:
    """Load a KYC document (image or PDF) as a BGR OpenCV image.

    The KYC OCR pipeline was written around ``cv2.imread`` which cannot
    read PDFs. Many uploaded Aadhaar / PAN cards arrive as single-page
    PDFs with a scanned image inside, so we render / extract them here.
    """
    if not os.path.exists(path):
        raise ValueError(f"Could not read document: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return _load_pdf_first_page_as_bgr(path, dpi)

    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def _load_pdf_first_page_as_bgr(path: str, dpi: int) -> np.ndarray:
    doc = fitz.open(path)

    try:
        if doc.page_count == 0:
            raise ValueError(f"PDF has no pages: {path}")

        page = doc[0]

        # Prefer the embedded image at native resolution when the page is
        # just a single scanned picture (sharper than re-rasterising).
        images = page.get_images()

        if len(images) == 1:
            embedded = _extract_embedded_image(doc, images[0][0])

            if embedded is not None:
                return embedded

        pix = page.get_pixmap(dpi=dpi)

        buffer = np.frombuffer(pix.samples, dtype=np.uint8)
        buffer = buffer.reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            return cv2.cvtColor(buffer, cv2.COLOR_RGBA2BGR)

        if pix.n == 1:
            return cv2.cvtColor(buffer, cv2.COLOR_GRAY2BGR)

        return cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)

    finally:
        doc.close()


def _extract_embedded_image(doc, xref):
    try:
        info = doc.extract_image(xref)
        data = np.frombuffer(info["image"], dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_document_to_temp_image(path: str, dpi: int = 300) -> str:
    """Return a filesystem path to an image for the given document.

    If ``path`` is already an image it is returned unchanged. If it is a
    PDF, the first page is rendered to a sibling ``.png`` and that path is
    returned, so existing helpers that take an image path keep working.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext != ".pdf":
        return path

    image = load_document_as_bgr(path, dpi)

    output_path = os.path.splitext(path)[0] + "_pdf_page1.png"
    cv2.imwrite(output_path, image)

    return output_path
