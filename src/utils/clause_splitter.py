"""Split contract text into clause-sized chunks for risk analysis."""

import re


def split_into_clauses(text: str) -> list[str]:
    chunks: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    for paragraph in paragraphs:
        if len(paragraph) > 400:
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", paragraph)
                if len(s.strip()) > 40
            ]
            chunks.extend(sentences or [paragraph])
        elif len(paragraph) > 30:
            chunks.append(paragraph)

    if not chunks:
        chunks = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s.strip()) > 40
        ]

    return chunks
