"""Write a fictional service-agreement PDF used as a demo input.

    Run from the repo root:
        python scripts/build_demo_contract.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "live_demo_service_agreement.pdf"

CLAUSES = [
    "SERVICE AGREEMENT",
    "This Service Agreement is made on 15 January 2026 at Jaipur, Rajasthan, between Aarohi Digital Private Limited, having its registered office at 42 Malviya Nagar, Jaipur 302017 (the Company), and Neelkanth Facilities LLP, having its office at C-18 Sitapura Industrial Area, Jaipur 302022 (the Vendor).",
    "1. Services. The Vendor shall provide on-site IT helpdesk and network maintenance services to the Company from 1 February 2026 until 31 January 2027. The service fee is Rs. 18,50,000 per year, payable monthly at Rs. 1,54,167, within 15 days of invoice.",
    "2. Confidentiality. Each party shall keep the other party's business information confidential for three years after this Agreement ends. Confidential information does not include information that is already public.",
    "3. Liability. The Vendor shall have unlimited liability for all losses, claims, and damages arising out of the services, whether direct, indirect, or consequential. There is no cap on liability and no exclusion for lost profits.",
    "4. Indemnity. The Vendor shall indemnify and hold harmless the Company against all third-party claims, costs, and legal fees. This indemnity is not mutual. The Company has no indemnity obligation to the Vendor.",
    "5. Non-compete and non-solicit. During this Agreement and for 24 months after it ends, the Vendor shall not provide similar IT services to any customer of the Company in Rajasthan, and shall not solicit the Company's employees or customers.",
    "6. Term and renewal. This Agreement renews automatically for successive one-year terms unless the Company gives notice. The Vendor may not terminate for convenience. The Company may terminate without cause on 7 days' written notice.",
    "7. Payment. If the Company disputes an invoice, payment may be withheld for 120 days. Late payment by the Company shall not incur interest.",
    "8. Intellectual property. All software, configurations, and work product created under this Agreement are assigned to the Company. The Vendor receives no licence to reuse that work.",
    "9. Limitation for the Company. The Company's total liability to the Vendor shall not exceed the fees paid in the previous one month.",
    "10. Governing law. This Agreement is governed by the laws of India. Courts at Jaipur shall have exclusive jurisdiction. Nothing in this Agreement creates a partnership or employment relationship.",
    "Signed for Aarohi Digital Private Limited and for Neelkanth Facilities LLP. This document is a fictional classroom sample for demonstration. It is not a signed contract and it is not legal advice.",
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
    )
    title = ParagraphStyle("Title", fontName="Times-Bold", fontSize=16, leading=20, spaceAfter=12)
    body = ParagraphStyle("Body", fontName="Times-Roman", fontSize=11, leading=16, spaceAfter=10)
    story = []
    for index, text in enumerate(CLAUSES):
        style = title if index == 0 else body
        story.append(Paragraph(text.replace("\n", "<br/>"), style))
        story.append(Spacer(1, 6))
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
