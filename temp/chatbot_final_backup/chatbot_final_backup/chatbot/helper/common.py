from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    UnstructuredImageLoader,
)
import re
# import cv2
# import pytesseract
# from PIL import Image
# from io import BytesIO
from docx import Document  # For extracting images from .docx files
import numpy as np
# from pdf2image import convert_from_path  # Convert PDF pages to images
from logger_config import setup_logger
import os
import traceback

logger = setup_logger()

# def extract_text_from_pdf_images(pdf_path):
#     """Extracts text from images in a PDF using OCR, handling errors gracefully."""
#     logger.info("Extracting images from PDF: %s", pdf_path)
#     extracted_text = ""
#
#     try:
#         # Convert PDF pages to images
#         images = convert_from_path(pdf_path, dpi=300)  # Higher DPI gives better OCR accuracy
#     except Exception as e:
#         logger.error(f"❌ Error converting PDF to images: {str(e)}")
#         return ""
#
#     if not images:
#         logger.warning("⚠️ No images found in the PDF.")
#         return ""
#
#     for i, image in enumerate(images):
#         try:
#             logger.info(f"Processing PDF page {i + 1}/{len(images)} as image")
#
#             # Convert PIL Image to OpenCV format (handle possible errors)
#             image_np = np.array(image)
#
#             # Convert image to grayscale for better OCR accuracy
#             gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
#
#             # Apply OCR
#             ocr_text = pytesseract.image_to_string(gray_image)
#
#             if ocr_text.strip():
#                 extracted_text += "\n" + ocr_text
#             else:
#                 logger.warning(f"⚠️ No text detected on page {i + 1}.")
#
#         except Exception as e:
#             logger.warning(f"⚠️ Skipping page {i + 1} due to an error: {str(e)}")
#
#     if not extracted_text.strip():
#         logger.info("⚠️ No valid text extracted from PDF images.")
#
#     logger.info("Extracted text from images in PDF: %s", extracted_text[:500])  # Preview first 500 chars
#     return extracted_text


# def extract_images_from_docx(file_path):
#     """Extracts images from a .docx file and applies OCR."""
#     logger.info("Extracting images from DOCX file: %s", file_path)
#     doc = Document(file_path)
#     extracted_text = ""
#     logger.info("Extracting images from DOCX begin")
#     for rel in doc.part.rels:
#         try:
#             if "image" in doc.part.rels[rel].target_ref:
#                 image_data = doc.part.rels[rel].target_part.blob
#
#                 # Convert BytesIO image data to PIL image
#                 image = Image.open(BytesIO(image_data))
#
#                 # Convert PIL image to OpenCV format (numpy array)
#                 image_np = np.array(image)
#
#                 # Convert image to grayscale for better OCR accuracy
#                 gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
#
#                 # Apply OCR
#                 ocr_text = pytesseract.image_to_string(gray_image)
#                 extracted_text += "\n" + ocr_text
#         except Exception as e:
#             logger.warning(f"⚠️ Skipping unsupported image format in DOCX: {str(e)}")
#
#     if not extracted_text:
#         logger.info("No valid images found for OCR in DOCX.")
#     logger.info("Extracting images from DOCX ended")
#     # logger.info("Extracted text from images in DOCX: %s", extracted_text[:500])  # Preview first 500 chars
#     return extracted_text

def clean_text(text):
    """Cleans the extracted text by removing excessive dots and unwanted characters."""
    text = re.sub(r'\.{3,}', ' ', text)  # Replace multiple dots with space
    text = re.sub(r'[\n\r]+', ' ', text)  # Remove excessive new lines
    text = re.sub(r' {2,}', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

def load_document(file_path):
    """Load text from a document (.docx or .pdf) using LangChain's Unstructured loaders."""
    extracted_text = ""
    try:
        file_extension = file_path.split(".")[-1].lower()
        logger.info("Loading document from: %s", file_path)
        if file_extension == "docx":
            if not os.access(file_path, os.R_OK):
                logger.error(f"File is not readable: {file_path}")
                raise PermissionError(f"File is not readable: {file_path}")
            logger.info("Using UnstructuredWordDocumentLoader for DOCX file.")
            loader = UnstructuredWordDocumentLoader(file_path, verbose=True)
            logger.info("Document successfully wraped.")
            try:
                documents = loader.load()
                logger.info("Document successfully loaded.")
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(f"Error while loading document: {str(e)}")
                raise
            extracted_text = " ".join([doc.page_content for doc in documents])
            logger.info(f'extracted content {extracted_text}')
            # logger.info("Extracted content from the doc.")
            # Extract images and apply OCR
            # extracted_text += extract_images_from_docx(file_path)
            logger.info("Extracted content from the image is done in main.")

        elif file_extension == "pdf":
            logger.info("Using UnstructuredPDFLoader for PDF file.")
            loader = UnstructuredPDFLoader(file_path)
            documents = loader.load()
            logger.info("Document successfully loaded.")
            extracted_text = " ".join([doc.page_content for doc in documents])
            # Extract images and apply OCR
            logger.info("Image extracted from the doc started.")
            # extracted_text += extract_text_from_pdf_images(file_path)
            logger.info("Image extracted from the doc ended.")

        elif file_extension in ["jpg", "jpeg", "png"]:
            logger.info("Using UnstructuredImageLoader for Image file.")
            loader = UnstructuredImageLoader(file_path, mode="elements", strategy="ocr_only")
            documents = loader.load()
            logger.info("Document successfully loaded.")
            extracted_text = " ".join([doc.page_content for doc in documents])
        else:
            return "ERROR_UNSUPPORTED_FILE: Unsupported file type."


        # logger.info("Extracted content strip")
        # Ensure extracted_text is always a string
        extracted_text = extracted_text.strip() if extracted_text else ""
        # logger.info("Extracted content after strip")
        if not extracted_text:
            logger.error("Extracted text is empty after processing.")
            return "ERROR_EMPTY_TEXT: Extracted text is empty after processing."

        #clean the data
        # logger.info("Extracted content before clean")
        cleaned_text = clean_text(extracted_text)
        # logger.info("Extracted content after clean")

        # print("\nExtracted Text Preview:\n", cleaned_text[:1000], "\n")
        return cleaned_text

    except Exception as e:
        logger.error("Error loading document: %s", str(e))
        return f"ERROR_LOADING_FAILED: {str(e)}"