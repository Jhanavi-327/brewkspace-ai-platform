import os
import uuid
import shutil
import traceback
import re
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.investigation import Investigation
from app.models.document import Document
from app.models.property_record import PropertyRecord
from app.services.kyc.pan_extractor import (
    extract_pan_card
)
from app.services.kyc.pan_gemini_extractor import (
    extract_pan_card_gemini
)
from app.services.kyc.aadhaar_extractor import (
    extract_aadhaar_card
)
from app.services.patta.patta_gemini_extractor import (
    extract_patta_book_gemini
)
from app.services.mr.mr_gemini_extractor import (
    extract_mutation_register_gemini
)
from app.services.rtc.rtc_gemini_extractor import (
    extract_rtc_gemini
)

from app.services.ocr_service import extract_text
from app.services.rtc_table_service import extract_rtc_table_layout
from app.services.rtc_cell_ocr_service import extract_cells_with_text
from app.services.rtc_cell_block_mapper import map_rtc_cells
from app.agents.classifier_agent import classify_document_text
from app.agents.rtc_block_extraction_agent import extract_rtc_blocks
from app.services.gemini_rtc_test_service import rtc_gemini_test
from app.services.kyc.aadhaar_mapper import map_aadhaar_fields
from app.services.kyc.pan_mapper import map_pan_fields
from app.services.kyc.pan_mapper import map_pan_fields

router = APIRouter()

UPLOAD_DIR = "uploaded_documents"

ALLOWED_EXTENSIONS = {
    "pdf": "PDF",
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "tiff": "TIFF",
    "tif": "TIFF",
}

MAX_FILE_SIZE = 50 * 1024 * 1024


@router.post("/{investigation_id}/documents")
def upload_document(
    investigation_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    investigation = (
        db.query(Investigation)
        .filter(
            Investigation.id == investigation_id,
            Investigation.organisation_id == current_user.organisation_id
        )
        .first()
    )

    if not investigation:
        raise HTTPException(
            status_code=404,
            detail="Investigation not found"
        )

    filename = file.filename

    if not filename or "." not in filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid file name"
        )

    extension = filename.split(".")[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Only PDF, JPEG, PNG, and TIFF are allowed."
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    document_id = str(uuid.uuid4())
    safe_filename = f"{document_id}_{filename}"
    storage_path = os.path.join(UPLOAD_DIR, safe_filename)

    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 50 MB limit"
        )

    with open(storage_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    document = Document(

         
        id=document_id,
        investigation_id=investigation.id,
        organisation_id=current_user.organisation_id,
        original_filename=filename,
        file_type=ALLOWED_EXTENSIONS[extension],
        size_bytes=file_size,
        storage_path=storage_path,
        classification=None,
        classification_confidence=None,
        extracted_fields=None,
        processing_status="UPLOADED"
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    return {
        "message": "Document uploaded successfully",
        "document_id": document.id,
        "filename": document.original_filename,
        "file_type": document.file_type,
        "size_bytes": document.size_bytes,
        "storage_path": document.storage_path,
        "processing_status": document.processing_status
    }


@router.get("/{investigation_id}/documents")
def list_documents(
    investigation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    investigation = (
        db.query(Investigation)
        .filter(
            Investigation.id == investigation_id,
            Investigation.organisation_id == current_user.organisation_id
        )
        .first()
    )

    if not investigation:
        raise HTTPException(
            status_code=404,
            detail="Investigation not found"
        )

    documents = (
        db.query(Document)
        .filter(
            Document.investigation_id == investigation_id,
            Document.organisation_id == current_user.organisation_id
        )
        .all()
    )

    return documents


@router.post("/documents/{document_id}/process")
def process_document(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    document = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.organisation_id == current_user.organisation_id
        )
        .first()
    )

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        extracted_text = extract_text(
            file_path=document.storage_path,
            file_type=document.file_type
        )

        classification_result = classify_document_text(extracted_text)

        structured_extraction = None
        fallback_extraction = None
        table_layout = None
        cells_with_text = []

        property_record_saved = False
        property_record_id = None
        completeness_percentage = None

        classification = classification_result["classification"]
        pan_extraction_details = None

        # --------------------------------------------------
        # AADHAAR FALLBACK CLASSIFICATION
        # --------------------------------------------------
        # Aadhaar fallback requires document-specific evidence.
        # A plausible crop name, date or number does not identify Aadhaar.
        if classification == "Unclassified":
            raw_upper = (extracted_text or "").upper()

            has_aadhaar_identity = bool(
                re.search(
                    r"\b(?:AADHAAR|AADHAR|UIDAI)\b",
                    raw_upper,
                )
                or "UNIQUE IDENTIFICATION AUTHORITY" in raw_upper
            )

            if has_aadhaar_identity:
                classification = "AADHAAR_CARD"
                classification_result = {
                    "classification": "AADHAAR_CARD",
                    "confidence": 0.90,
                    "candidates": [
                        {
                            "type": "AADHAAR_CARD",
                            "confidence": 0.90,
                        }
                    ],
                }

        # --------------------------------------------------
        # RTC PROCESSING
        #
        # Karnataka RTC / Pahani (Village Account Form No. 2). Extracted
        # directly with Gemini vision, grounded by the pipeline OCR text (same
        # multi-model quota fallback as PAN / Aadhaar / Patta / MR).
        #
        # Gemini is PRIMARY because the Tesseract cell-OCR path below cannot
        # read this form: table detection merges the 13 blocks into a few
        # mega-cells and Kannada recognition substitutes Latin glyphs (ಕೋಂ ->
        # "BOLO"), which made the mapper emit printed form headings as values.
        # That path is kept as the fallback for when no Gemini model answers
        # (e.g. an exhausted daily quota), so an RTC still yields something.
        # --------------------------------------------------
        if classification == "RTC":

            fallback_extraction = extract_rtc_blocks(extracted_text)
            structured_extraction = fallback_extraction

            gemini_extraction = None

            try:
                gemini_extraction = extract_rtc_gemini(
                    document.storage_path,
                    ocr_text=extracted_text
                )
            except Exception as e:
                print("Gemini RTC extraction failed:", e)

            gemini_usable = bool(
                gemini_extraction
                and gemini_extraction.get("extraction_source")
                != "gemini_vision_rtc_v1:unavailable"
            )

            if gemini_usable:
                structured_extraction = gemini_extraction

            # Only pay for the slow cell-OCR sweep (hundreds of Tesseract calls
            # per page) when Gemini could not read the document at all.
            #
            # PDFs only, and that is a measured choice rather than a technical
            # limit — the renderer opens an image as readily as a PDF page, and
            # a screenshot of the Yalahanka RTC does yield 63 cells across 15
            # rows. But a screenshot is captured at display resolution and the
            # page-level OCR reads it well, while the per-cell crops of that
            # same image read it badly. Routing that screenshot through the
            # cells fixed the header (ಯಲಹಂಕ / ಕುದುರೆಗೆರೆ instead of whole raw
            # lines) and broke three other fields: survey 59 -> 29, extent
            # 78.04.00.00 -> 38.04.00.00, and three named owners -> "ಮ್".
            # Scans need the grid; screenshots do not.
            elif document.file_type == "PDF":
                try:
                    table_layout = extract_rtc_table_layout(document.storage_path)

                    cells_with_text = extract_cells_with_text(
                        document.storage_path,
                        table_layout
                    )

                    # The taluk/hobli/village line is printed above the
                    # table border, so it is in no cell and whole-page OCR is
                    # its only source — a pass the table drowns out on a
                    # watermarked scan. Read that strip on its own as well.
                    from app.services.rtc_cell_ocr_service import (
                        ocr_header_strip
                    )

                    header_strip_text = ocr_header_strip(
                        document.storage_path,
                        cells_with_text
                    )

                    structured_extraction = map_rtc_cells(
                        cells_with_text=cells_with_text,
                        table_layout=table_layout,
                        fallback_extraction=fallback_extraction,
                        raw_text=extracted_text,
                        header_strip_text=header_strip_text
                    )

                except Exception as table_error:
                    print("\n========== RTC MAPPER ERROR ==========")
                    traceback.print_exc()
                    print("======================================\n")

                    table_layout = {
                        "error": str(table_error),
                        "message": "Table layout / cell OCR extraction failed"
                    }

                    cells_with_text = []
                    structured_extraction = fallback_extraction

            # Carry the reason forward so the caller can tell a degraded read
            # from a confident one instead of silently trusting cell OCR.
            if not gemini_usable and isinstance(structured_extraction, dict):
                structured_extraction.setdefault("warnings", []).append(
                    "Gemini vision was unavailable for this RTC; values come"
                    " from the Tesseract cell-OCR fallback and need review."
                    + (
                        f" Gemini error: {gemini_extraction.get('error')}"
                        if gemini_extraction and gemini_extraction.get("error")
                        else ""
                    )
                )

        # --------------------------------------------------
        # PAN CARD PROCESSING
        # --------------------------------------------------
        elif classification == "PAN_CARD":

            enhanced_text = ""

            try:
                from app.services.kyc.pan_image_ocr import run_pan_ocr
                enhanced_text = run_pan_ocr(document.storage_path)

            except Exception as e:
                print("Enhanced PAN OCR failed:", e)

            combined_pan_text = (
                extracted_text
                + "\n\n--- ENHANCED PAN OCR ---\n"
                + enhanced_text
            )

            structured_extraction = None

            try:
                structured_extraction = extract_pan_card_gemini(
                    document.storage_path
                )

            except Exception as e:
                print("Gemini PAN extraction failed:", e)

            if not structured_extraction:
                # Pass the image so the DOB can be re-read with a digits-only
                # whitelist — full-page OCR misreads a single digit with no
                # disagreement to flag it (28/10/1958 -> 28/10/1956).
                structured_extraction = extract_pan_card(
                    combined_pan_text,
                    image_path=document.storage_path
                )
            pan_extraction_details = structured_extraction

            structured_extraction = map_pan_fields(
                pan_extraction_details or {}
            ).model_dump()
        # --------------------------------------------------
        # AADHAAR CARD PROCESSING
        # --------------------------------------------------
        elif classification == "AADHAAR_CARD":

            crop_lines = ""
            enhanced_text = ""

            # -----------------------------
            # Crop OCR
            # -----------------------------
            try:
                from app.services.kyc.aadhaar_crop_ocr import (
                    extract_aadhaar_fields_from_image
                )

                crop_fields = extract_aadhaar_fields_from_image(
                    document.storage_path
                )

                crop_lines = f"""
        --- AADHAAR FIELD CROP OCR ---
        AADHAAR_NUMBER: {crop_fields.get("aadhaar_number", "")}
        NAME: {crop_fields.get("name", "")}
        DOB: {crop_fields.get("dob", "")}
        GENDER: {crop_fields.get("gender", "")}
        """

            except Exception as e:
                print("Aadhaar crop OCR failed:", e)

            # -----------------------------
            # Enhanced OCR
            # -----------------------------
            try:
                from app.services.kyc.aadhaar_image_ocr import (
                    run_aadhaar_ocr
                )

                enhanced_text = run_aadhaar_ocr(
                    document.storage_path
                )

            except Exception as e:
                print("Enhanced Aadhaar OCR failed:", e)

            combined_aadhaar_text = (
                extracted_text
                + "\n\n"
                + crop_lines
                + "\n\n--- ENHANCED AADHAAR OCR ---\n"
                + enhanced_text
            )

            from app.services.kyc.aadhaar_extractor import (
                extract_aadhaar_card
            )

            # Pass the file so the printed QR can be decoded. It carries the
            # same fields as exact text, filling what OCR could not read and
            # cross-checking what it could — see aadhaar_qr for why it never
            # overwrites a printed value.
            structured_extraction = extract_aadhaar_card(
                combined_aadhaar_text,
                image_path=document.storage_path
            )
        # --------------------------------------------------
        # PATTA BOOK / KHATA EXTRACT PROCESSING
        #
        # Karnataka Khata Extract / Patta Book copy. This is a clean tabular
        # document, so we extract it directly with Gemini vision (same multi
        # model quota fallback as PAN / Aadhaar). Aadhaar and PAN handling
        # above is untouched.
        # --------------------------------------------------
        elif classification == "PATTA_BOOK":

            try:
                structured_extraction = extract_patta_book_gemini(
                    document.storage_path,
                    ocr_text=extracted_text
                )
            except Exception as e:
                print("Gemini Patta Book extraction failed:", e)
                structured_extraction = None

        # --------------------------------------------------
        # MUTATION REGISTER PROCESSING
        #
        # Karnataka ನಮೂನೆ-12 / ಮ್ಯುಟೇಶನ್ ರಿಜಿಸ್ಟರ್ ಪ್ರತಿ (Rule 46). Extracted
        # directly with Gemini vision, grounded by the pipeline OCR text (same
        # multi-model quota fallback as PAN / Aadhaar / Patta). Unlike Patta it
        # has no printed arithmetic total, so trust comes from a second-model
        # text-agreement pass instead of a numeric checksum. RTC / PAN / Aadhaar
        # / Patta handling is untouched.
        # --------------------------------------------------
        elif classification == "MUTATION_REGISTER":

            try:
                structured_extraction = extract_mutation_register_gemini(
                    document.storage_path,
                    ocr_text=extracted_text
                )
            except Exception as e:
                print("Gemini Mutation Register extraction failed:", e)
                structured_extraction = None

        # Check every Aadhaar visually, even when OCR fields are non-empty.
        aadhaar_extraction_details = None
        aadhaar_review_required = None

        if classification == "AADHAAR_CARD":
            from app.services.kyc.aadhaar_mapper import map_aadhaar_fields

            ocr_extraction = structured_extraction or {}
            gemini_extraction = None
            vision_error = None

            try:
                from app.services.kyc.aadhaar_gemini_extractor import (
                    extract_aadhaar_card_gemini
                )

                gemini_extraction = extract_aadhaar_card_gemini(
                    document.storage_path
                )
            except Exception as error:
                vision_error = type(error).__name__
                print("Gemini Aadhaar extraction failed:", error)

            fields = (
                "name",
                "date_of_birth_or_year",
                "gender",
                "aadhaar_number",
                "address",
            )

            vision_usable = (
                isinstance(gemini_extraction, dict)
                and gemini_extraction.get("document_type") == "AADHAAR_CARD"
                and any(gemini_extraction.get(field) for field in fields)
            )

            disagreements = []

            if vision_usable:
                for field in fields:
                    ocr_value = ocr_extraction.get(field)
                    vision_value = gemini_extraction.get(field)

                    if ocr_value and vision_value:
                        left = " ".join(str(ocr_value).split()).casefold()
                        right = " ".join(str(vision_value).split()).casefold()

                        if left != right:
                            disagreements.append({
                                "field": field,
                                "ocr": ocr_value,
                                "vision": vision_value,
                            })

                # Missing visual fields stay null; do not silently fill them
                # from the OCR path whose readings may be incorrect.
                selected = gemini_extraction
            else:
                selected = {}
                vision_error = vision_error or "No usable visual extraction"

            structured_extraction = map_aadhaar_fields(
                selected
            ).model_dump()

            aadhaar_review_required = (
                not vision_usable
                or bool(disagreements)
                or bool(ocr_extraction.get("qr_disagreements"))
                or any(
                    value is None
                    for value in structured_extraction.values()
                )
            )

            aadhaar_extraction_details = {
                "ocr": ocr_extraction,
                "vision": gemini_extraction,
                "selected_source": "vision" if vision_usable else None,
                "disagreements": disagreements,
                "vision_error": vision_error,
                "review_required": aadhaar_review_required,
            }

        # --------------------------------------------------
        # PROPERTY RECORD SAVE ONLY FOR RTC
        # --------------------------------------------------
        if structured_extraction and classification == "RTC":

            header = structured_extraction.get("header", {})
            survey_block = structured_extraction.get("1_survey_number", {})
            hissa_block = structured_extraction.get("2_hissa", {})
            khata_block = structured_extraction.get("3_khata_vara", {})
            owner_block = structured_extraction.get("9_ownership_details", [])

            property_record = (
                db.query(PropertyRecord)
                .filter(
                    PropertyRecord.investigation_id == document.investigation_id,
                    PropertyRecord.organisation_id == current_user.organisation_id
                )
                .first()
            )

            if not property_record:
                property_record = PropertyRecord(
                    id=str(uuid.uuid4()),
                    investigation_id=document.investigation_id,
                    organisation_id=current_user.organisation_id
                )
                db.add(property_record)

            property_record.taluk = header.get("taluk")
            property_record.hobli = header.get("hobli")
            property_record.village = header.get("village")
            property_record.survey_number = survey_block.get("survey_number")

            property_record.fields = {
                "rtc_blocks": structured_extraction,
                "table_layout": table_layout,
                "cells_with_text": cells_with_text
            }

            required_fields = [
                header.get("taluk"),
                header.get("hobli"),
                header.get("village"),
                survey_block.get("survey_number"),
                hissa_block.get("hissa_number"),
                khata_block.get("total_extent"),
                owner_block
            ]

            filled_count = sum(1 for value in required_fields if value)

            completeness_percentage = round(
                (filled_count / len(required_fields)) * 100
            )

            property_record.completeness_percentage = completeness_percentage

            property_record_saved = True
            property_record_id = property_record.id

        # Map AFTER the property-record code, which uses numbered RTC blocks.
        rtc_extraction_details = None
        rtc_review_required = None
        if classification == "RTC":
            from app.services.rtc.rtc_mapper import map_rtc_fields

            rtc_extraction_details = structured_extraction or {}
            structured_extraction = map_rtc_fields(
                rtc_extraction_details
            ).model_dump()
            verification = rtc_extraction_details.get("text_verification") or {}
            differences = rtc_extraction_details.get("disagreements") or {}
            rtc_review_required = bool(
                rtc_extraction_details.get("warnings")
                or rtc_extraction_details.get("error")
                or rtc_extraction_details.get("extraction_status") != "extracted"
                or verification.get("status") == "disagreement"
                or any(differences.values())
            )
            # Unassigned/document-wide entries stay in extraction evidence.
            unassigned_rights = rtc_extraction_details.get("11_rights_liabilities") or {}
            unassigned_possession = rtc_extraction_details.get("10_possession_details") or {}
            if any(unassigned_rights.values()) or any(unassigned_possession.values()):
                rtc_review_required = True
                rtc_extraction_details.setdefault("warnings", []).append(
                    "Some possession or rights information is document-wide or "
                    "could not be assigned to an owner; inspect rtc_extraction_details."
                )

            if property_record_saved:
                property_record.fields = {
                    **property_record.fields,
                    "rtc_schema": structured_extraction,
                }

        # Normalize MR after extraction; keep detailed evidence outside the schema.
        mr_extraction_details = None
        mr_review_required = None
        mr_mapping_warnings = []
        if classification == "MUTATION_REGISTER":
            from app.services.mr.mr_mapper import map_mr_fields, mr_requires_review

            mr_extraction_details = structured_extraction or {}
            mr_model = map_mr_fields(mr_extraction_details, mr_mapping_warnings)
            mr_review_required = mr_requires_review(
                mr_extraction_details, mr_model, mr_mapping_warnings
            )
            mr_extraction_details = {
                **mr_extraction_details,
                "schema_mapping_warnings": mr_mapping_warnings,
            }
            structured_extraction = mr_model.model_dump()

        document.classification = classification_result["classification"]
        document.classification_confidence = classification_result["confidence"]

        document.extracted_fields = {
            "raw_text_preview": extracted_text[:5000],
            "classification_candidates": classification_result["candidates"],
            "structured_extraction": structured_extraction,
            "table_layout": table_layout,
            "cells_with_text": cells_with_text,
            "fallback_regex_extraction": fallback_extraction
        }

        if classification == "AADHAAR_CARD":
            document.extracted_fields = {
                **document.extracted_fields,
                "aadhaar_extraction_details": aadhaar_extraction_details,
            }

        if classification == "PAN_CARD":
            document.extracted_fields = {
                **document.extracted_fields,
                "pan_extraction_details": pan_extraction_details,
            }

        if classification == "RTC":
            document.extracted_fields = {
                **document.extracted_fields,
                "rtc_extraction_details": rtc_extraction_details,
            }

        if classification == "MUTATION_REGISTER":
            document.extracted_fields = {
                **document.extracted_fields,
                "mr_extraction_details": mr_extraction_details,
            }

        document.processing_status = "PROCESSED"

        db.commit()
        db.refresh(document)

        response = {
            "message": "Document processed successfully",
            "document_id": document.id,
            "classification": document.classification,
            "confidence": document.classification_confidence,
            "candidates": classification_result["candidates"],
            "structured_extraction": structured_extraction,
            "table_layout": table_layout,
            "cells_with_text": cells_with_text,
            "property_record_saved": property_record_saved,
            "property_record_id": property_record_id,
            "completeness_percentage": completeness_percentage,
            "raw_text_preview": extracted_text[:3000]
        }

        if classification == "AADHAAR_CARD":
            response["review_required"] = aadhaar_review_required
            if aadhaar_review_required:
                response["message"] = "Document processed; extraction requires review"

        if classification == "RTC":
            response["review_required"] = rtc_review_required
            response["extraction_warnings"] = rtc_extraction_details.get("warnings", [])
            if rtc_review_required:
                response["message"] = "Document processed; extraction requires review"

        if classification == "MUTATION_REGISTER":
            response["review_required"] = mr_review_required
            response["extraction_warnings"] = list(dict.fromkeys(
                mr_mapping_warnings
                + (mr_extraction_details.get("mapping_warnings") or [])
            ))
            if mr_review_required:
                response["message"] = "Document processed; extraction requires review"

        return response

    except Exception as e:
        db.rollback()
        document.processing_status = "FAILED"
        db.commit()

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(e)}"
        )




    
@router.post("/documents/{document_id}/test-gemini")
def test_gemini(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    document = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.organisation_id == current_user.organisation_id
        )
        .first()
    )

    if not document:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    try:
        result = rtc_gemini_test(
            document.storage_path
        )

        return {
            "message": "Gemini RTC test completed",
            "document_id": document.id,
            "filename": document.original_filename,
            "gemini_result": result
        }

    except Exception as e:
        return {
            "message": "Gemini RTC test failed",
            "document_id": document.id,
            "error_type": type(e).__name__,
            "error": str(e)
        }