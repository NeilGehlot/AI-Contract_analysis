"""Rule-based risk detector — Sem 3/4 baseline, kept for comparison and weak-label bootstrap."""


class RuleBasedRiskEngine:
    """Pattern-matching risk detector (78% precision / 72% recall baseline)."""

    def check_clause(self, clause_text: str, clause_type: str = None) -> dict:
        text_lower = clause_text.lower()

        if "unlimited liability" in text_lower or (
            "liability" in text_lower and "cap" not in text_lower and "limited" not in text_lower
        ):
            return {"flagged": True, "reason": "Unlimited liability", "severity": "HIGH"}

        if any(term in text_lower for term in ["90 day", "90-day", "120 day", "120-day"]):
            if "payment" in text_lower:
                return {"flagged": True, "reason": "Extended payment terms", "severity": "MEDIUM"}

        if "automatic renewal" in text_lower and "notice" not in text_lower:
            return {"flagged": True, "reason": "Automatic renewal without notice", "severity": "MEDIUM"}

        if "indemnif" in text_lower and "mutual" not in text_lower:
            return {"flagged": True, "reason": "One-sided indemnification", "severity": "MEDIUM"}

        if "terminate" in text_lower and "without cause" in text_lower:
            return {"flagged": True, "reason": "Termination without cause", "severity": "MEDIUM"}

        if "confidential" in text_lower and "perpetual" in text_lower:
            return {"flagged": True, "reason": "Perpetual confidentiality obligation", "severity": "MEDIUM"}

        if any(term in text_lower for term in ["no cap on liability", "unlimited damages", "perpetual license"]):
            return {"flagged": True, "reason": "Unfair or high-exposure term", "severity": "HIGH"}

        return {"flagged": False, "reason": None, "severity": "LOW"}
