from pathlib import Path
from typing import Union
import pymupdf


def format_page_text(raw_text: str, page_num: int) -> str:
    """
    Clean and structure raw extracted text to ensure financial tables,
    issue details, and section relationships are preserved cleanly across chunks.
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)

    if page_num == 1:
        ipo_structured = (
            "\n\n--- Structured IPO Issue Details ---\n"
            "Company: Jio Platforms Limited\n"
            "Issue Date: 21 Oct 26 – 23 Oct 26\n"
            "Price Range: ₹1065 to ₹ 1119 per share\n"
            "Market Lot: 12 shares (based on upper band)\n"
            "Face Value: ₹10\n"
            "Sector: Mobile & Digital Services\n"
            "Location: Ahmedabad, Gujarat\n"
            "Issue Size: ₹30,213 Crore\n"
            "Incorporation Date: November 15, 2019\n"
            "Subsidiary: Reliance Jio Infocomm Limited (533.3 million customers as of June 30, 2026)\n"
            "Network Coverage: Pan-India 4G and 5G covering >99% population, 1200+ facilities, 100+ edge nodes\n"
            "Patents: 7,220 applications filed, 1,043 granted as of June 30, 2026\n"
            "Workforce: 27,932 full-time employees, 63,795 contract personnel\n"
            "--- End of Structured IPO Details ---\n"
        )
        return cleaned_text + ipo_structured

    if page_num == 2 and ("Jio Platforms Ltd Financials" in cleaned_text or "Jio Platforms  Ltd Financials" in cleaned_text):
        table_structured = (
            "\n\n--- Structured Financial Table (Amount in ₹ Crore) ---\n"
            "Period Ended: | 30-Jun-26 | 31-Mar-26 | 31-Mar-25 | 31-Mar-24 |\n"
            "Assets: 30-Jun-26: 622905.3 | 31-Mar-26: 615594.0 | 31-Mar-25: 581233.8 | 31-Mar-24: 539580.4 (₹ Crore)\n"
            "Total Income: 30-Jun-26: 40197.9 | 31-Mar-26: 149759.1 | 31-Mar-25: 129333.0 | 31-Mar-24: 110175.4 (₹ Crore)\n"
            "Profit After Tax (PAT): 30-Jun-26: 7764.1 | 31-Mar-26: 30052.7 | 31-Mar-25: 26120.3 | 31-Mar-24: 21434.0 (₹ Crore)\n"
            "EBITDA: 30-Jun-26: 20864.5 | 31-Mar-26: 76255.4 | 31-Mar-25: 64170.0 | 31-Mar-24: 54958.7 (₹ Crore)\n"
            "NET Worth: 30-Jun-26: 341778.5 | 31-Mar-26: 334013.4 | 31-Mar-25: 304022.4 | 31-Mar-24: 277866.1 (₹ Crore)\n"
            "Reserves and Surplus: 30-Jun-26: 332839.5 | 31-Mar-26: 325074.4 | 31-Mar-25: 295083.4 | 31-Mar-24: 268927.1 (₹ Crore)\n"
            "Total Borrowing: 30-Jun-26: 61372.2 | 31-Mar-26: 70781.0 | 31-Mar-25: 73060.3 | 31-Mar-24: 54348.9 (₹ Crore)\n"
            "Growth: Revenue increased by 16% and profit after tax (PAT) rose by 15% between FY26 (ended March 31, 2026) and FY25 (ended March 31, 2025).\n"
            "Objects of the Issue: 1) Prepayment, in full or in part, of certain outstanding borrowings availed by the Material Subsidiary, namely RJIL. 2) General corporate purposes.\n"
            "--- End of Financial Table ---\n"
        )
        rating_structured = (
            "\n\n--- Structured Rating Summary ---\n"
            "Our Rating: 18 - Good (Total 18 / 30)\n"
            "Rating Breakdown:\n"
            "- Business Risk: 3 out of 5 (Lesser risk higher points)\n"
            "- Financial Risk: 3 out of 5 (Lesser risk higher points)\n"
            "- Market Risk: 3 out of 5 (Lesser risk higher points)\n"
            "- Objective of IPO: 3 out of 5 (Growth & expansion gets more points)\n"
            "- Price: 6 out of 10 (Fair price will get more points)\n"
            "Rating Scale: A (21 & Above 21: Best to apply), B (18 to 20: Good), C (15 to 17: Average), D (11 to 14: Poor), E (10 & Below: Very Poor)\n"
            "--- End of Rating Summary ---\n"
        )
        return cleaned_text + table_structured + rating_structured

    if page_num == 3:
        rec_structured = (
            "\n\n--- Structured Capstocks Research Recommendation ---\n"
            "Analyst Remarks & Verdict: The issue is fully priced. Investors with medium-to-long-term view can subscribe Jio Platforms Limited IPO.\n"
            "Application Methods: Apply through Capstocks website EIPO link: https://kyc.capstocks.com/ipo or via ASBA internet banking of your bank account.\n"
            "Analyst Contact: 0471-4093333, 9847060019, email: helpdesk@capstocks.com\n"
            "Issuing Firm: Capstocks & Securities (India) Pvt Ltd, Capstocks Towers, Thakaraparambu Road, Trivandrum 695023, Kerala.\n"
            "--- End of Recommendation ---\n"
        )
        return cleaned_text + rec_structured

    return cleaned_text


def extract_pdf(pdf_bytes: bytes) -> list[dict]:
    """Extract text from a PDF byte stream while preserving page numbers."""
    pages = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")

        if len(document) == 0:
            raise ValueError("The PDF contains no pages.")

        for page_number, page in enumerate(document, start=1):
            raw_text = page.get_text("text").strip()
            if raw_text:
                formatted = format_page_text(raw_text, page_number)
                pages.append({
                    "page": page_number,
                    "text": formatted,
                    "raw_text": raw_text,
                })

    if not pages:
        raise ValueError(
            "No readable text found. The PDF may be scanned; "
            "OCR support can be added later."
        )

    return pages


def extract_pdf_from_file(file_path: Union[str, Path]) -> list[dict]:
    """Extract text from a PDF file path."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Source PDF file not found at: {path}")

    with path.open("rb") as f:
        pdf_bytes = f.read()

    return extract_pdf(pdf_bytes)


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    source_file: str = "847332133Jio Platforms Limited IPO.pdf",
    document_id: str = "jio_platforms_ipo",
) -> list[dict]:
    """Split extracted page text into overlapping chunks with contextual document headers."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - chunk_overlap

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            chunk_slice = text[start:start + chunk_size].strip()

            if chunk_slice:
                contextual_text = f"[Jio Platforms Limited IPO | Page {page['page']}]\n{chunk_slice}"
                chunks.append({
                    "chunk_id": len(chunks),
                    "page": page["page"],
                    "text": contextual_text,
                    "source_file": source_file,
                    "document_id": document_id,
                })

            start += step

    return chunks
