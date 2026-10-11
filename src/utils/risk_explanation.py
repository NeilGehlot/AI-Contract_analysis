"""Turn a flagged clause into a short, checkable reason."""

from __future__ import annotations


def explain_clause(clause: str, severity: str) -> dict:
    """Return what the wording is about, and whether that severity fits."""
    text = clause.lower()
    topic, fit, action = _match(text, severity)
    return {
        "topic": topic,
        "consideration": fit,
        "recommendation": action,
    }


def _match(text: str, severity: str) -> tuple[str, str, str]:
    if any(p in text for p in ("unlimited liability", "no cap on liability", "lost profits")):
        return (
            "Uncapped liability",
            "This removes any ceiling on what the Vendor must pay, including indirect loss. HIGH is the right call.",
            "Ask for a liability cap, and exclude indirect and consequential damages.",
        )
    if "indemn" in text or "hold harmless" in text:
        return (
            "One-sided indemnity",
            "Only the Vendor covers third-party claims. A mutual indemnity would be more balanced. MEDIUM or HIGH both fit.",
            "Make the indemnity mutual, and limit it to losses caused by the Vendor.",
        )
    if "non-compete" in text or ("solicit" in text) or "24 months" in text:
        return (
            "Non-compete and non-solicit",
            "The Vendor is blocked from similar work and from approaching the Company's staff or customers for 24 months. That is a real restraint. HIGH is a fair reading; the model may call it MEDIUM.",
            "Shorten the period and narrow the geography and the kind of work covered.",
        )
    if "automatic" in text or "renews automatically" in text:
        return (
            "Automatic renewal",
            "The contract rolls on unless the Company gives notice. The Vendor has no matching exit. MEDIUM is appropriate.",
            "Require notice from both sides, and allow either party to stop a renewal.",
        )
    if "terminate" in text and ("without cause" in text or "may not terminate" in text or "7 days" in text):
        return (
            "One-sided termination",
            "The Company can walk away quickly, or the Vendor cannot leave at all. That imbalance is a real commercial risk.",
            "Give both parties the same right to end the contract, with a longer notice period.",
        )
    if "120 day" in text or "120 days" in text:
        return (
            "Long payment hold",
            "A disputed invoice can sit unpaid for 120 days. That is well beyond a normal 15- to 30-day cycle. HIGH or MEDIUM both fit.",
            "Cut the dispute hold to 30 days and require interest on undisputed late amounts.",
        )
    if "shall not incur interest" in text or "not incur interest" in text:
        return (
            "No interest on late payment",
            "The Company can pay late without a cost. That weakens the Vendor's payment right. MEDIUM is reasonable.",
            "Add interest on undisputed amounts that are paid late.",
        )
    if "assigned to the company" in text or "work product" in text:
        return (
            "IP assignment",
            "Everything created under the contract is handed to the Company. That is common, but it is one-way. MEDIUM is a fair label.",
            "Keep the Vendor's pre-existing tools, and grant the Company a licence instead of a full assignment where possible.",
        )
    if "no licence" in text or "no license" in text:
        return (
            "No right to reuse the work",
            "The Vendor cannot reuse what it built. That is strict, but it follows from the assignment. HIGH is harsher than a typical review.",
            "Allow reuse of generic tools and know-how that are not specific to the Company.",
        )
    if "shall not exceed" in text or "liability to the vendor" in text:
        return (
            "Liability cap that protects only the Company",
            "A cap is usually protective. Here it applies only to the Company, and only to one month of fees, while the Vendor's liability is uncapped. The imbalance is the issue, not the existence of a cap.",
            "Use the same cap for both parties.",
        )
    if "governed by" in text or "exclusive jurisdiction" in text:
        return (
            "Governing law and courts",
            f"The model marked this {severity}, but the sentence only chooses a governing law or a court city. That is routine. It is not a high commercial risk.",
            "No change needed unless the parties want a different city or arbitration.",
        )
    if "partnership" in text or "employment relationship" in text or "not a signed" in text or "classroom sample" in text or "legal advice" in text or text.startswith("signed for"):
        return (
            "Boilerplate or demo note",
            f"The model marked this {severity}. The text does not create a commercial exposure. This flag is not correct.",
            "Ignore this item in the review.",
        )
    if "service fee" in text or "payable monthly" in text or "helpdesk" in text:
        return (
            "Scope or price",
            f"The model marked this {severity}. The sentence states the work or a normal 15-day payment term. It is not, by itself, an unfair clause.",
            "Treat this as background, not as a risk, unless the fee or scope is wrong.",
        )
    return (
        "Model score without a clear risk phrase",
        f"The model marked this {severity} from the wording pattern. No standard high-risk phrase (uncapped liability, one-sided indemnity, non-compete, auto-renewal, or a long payment hold) is present, so check the sentence before treating it as a real risk.",
        "Read the clause. Negotiate only if the obligation is actually one-sided.",
    )
