import re

from schemas import SplitFindingsOutput


def normalize_for_span_check(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_output(raw_obj: dict, original_findings: str) -> SplitFindingsOutput:
    parsed = SplitFindingsOutput.model_validate(raw_obj)

    original_norm = normalize_for_span_check(original_findings)

    for item in parsed.findings:
        span_norm = normalize_for_span_check(item.source_span)

        if span_norm not in original_norm:
            raise ValueError(
                f"source_span not found in original findings: {item.source_span}"
            )

        if not item.sentence.endswith("."):
            raise ValueError(f"Sentence does not end with period: {item.sentence}")

        if len(item.sentence) > 300:
            raise ValueError(f"Sentence too long: {item.sentence}")

    return parsed