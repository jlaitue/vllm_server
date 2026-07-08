import re
from difflib import SequenceMatcher

from schemas import SplitFindingsOutput


NEGATION_CUES = [
    "no",
    "without",
    "negative for",
    "free of",
    "absence of",
]


TOPIC_KEYWORDS = {
    "disease-atelectasis": ["atelectasis", "atelectatic"],
    "disease-cardiomegaly": ["cardiomegaly", "cardiomegalic"],
    "disease-consolidation": ["consolidation", "consolidative"],
    "disease-edema": ["edema", "oedema", "vascular congestion", "pulmonary vascular congestion", "congestion"],
    "disease-enlarged cardiomediastinum": [
        "enlarged cardiomediastinal",
        "cardiomediastinal silhouette is enlarged",
        "widened cardiomediastinal",
    ],
    "disease-fracture": ["fracture", "fractures"],
    "disease-lung lesion": ["nodule", "nodules", "mass", "lesion", "lesions"],
    "disease-lung opacity": ["opacity", "opacities", "airspace opacity", "airspace opacities"],
    "disease-pleural effusion": ["pleural effusion", "effusion", "effusions"],
    "disease-pleural other": ["pleural thickening", "pleural plaque", "pleural plaques"],
    "disease-pneumonia": ["pneumonia", "infection"],
    "disease-pneumothorax": ["pneumothorax", "pneumothoraces"],
    "organ-heart": ["heart", "cardiac silhouette", "cardiac size"],
    "organ-lungs": ["lungs", "lung"],
    "organ-pleura": ["pleura", "pleural surfaces"],
    "organ-mediastinum": ["mediastinum", "mediastinal", "hilar", "hila", "aorta", "aortic"],
    "organ-bones": ["osseous", "bone", "bones", "rib", "ribs", "spine", "thoracic spine", "clavicle", "clavicles"],
    "organ-diaphragm": ["diaphragm", "hemidiaphragm", "subdiaphragmatic"],
    "support devices": [
        "tube", "line", "catheter", "port", "pacemaker", "icd", "lead", "leads",
        "sternotomy wires", "clips", "chest tube", "drain",
    ],
    "patient status": ["low volume", "low-volume", "portable", "ap view", "pa and lateral"],
    "other": [],
}


def normalize_for_span_check(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_clauses_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """
    Split into sentence-like chunks while preserving character offsets.
    We keep this simple because radiology findings are usually sentence-based.
    """
    chunks: list[tuple[str, int, int]] = []
    start = 0

    for match in re.finditer(r"[.;]", text):
        end = match.end()
        chunk = text[start:end].strip()
        leading_ws = len(text[start:end]) - len(text[start:end].lstrip())

        if chunk:
            chunk_start = start + leading_ws
            chunk_end = end
            chunks.append((text[chunk_start:chunk_end].strip(), chunk_start, chunk_end))

        start = end

    if start < len(text):
        chunk_raw = text[start:]
        chunk = chunk_raw.strip()
        leading_ws = len(chunk_raw) - len(chunk_raw.lstrip())

        if chunk:
            chunk_start = start + leading_ws
            chunk_end = len(text)
            chunks.append((text[chunk_start:chunk_end].strip(), chunk_start, chunk_end))

    return chunks


def exact_or_normalized_span_found(source_span: str, original_findings: str) -> bool:
    if source_span in original_findings:
        return True

    span_norm = normalize_for_span_check(source_span)
    original_norm = normalize_for_span_check(original_findings)
    return span_norm in original_norm


def get_topic_keywords(topic: str) -> list[str]:
    return TOPIC_KEYWORDS.get(topic, [])


def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    text_norm = normalize_for_span_check(text)
    return any(keyword in text_norm for keyword in keywords)


def contains_negation_cue(text: str) -> bool:
    text_norm = normalize_for_span_check(text)
    return any(re.search(rf"\b{re.escape(cue)}\b", text_norm) for cue in NEGATION_CUES)


def repair_source_span(
    source_span: str,
    original_findings: str,
    topic: str,
    polarity: str,
) -> str | None:
    """
    Attempt to repair common LLM source_span mistakes.

    Example:
    original: "There is no pleural effusion or pneumothorax."
    bad span: "no pneumothorax"
    repaired: "no pleural effusion or pneumothorax"

    Returns a repaired exact substring from original_findings, or None.
    """
    if exact_or_normalized_span_found(source_span, original_findings):
        return source_span

    source_norm = normalize_for_span_check(source_span)
    keywords = get_topic_keywords(topic)

    # First try sentence/clause-level repair.
    candidates: list[str] = []
    for chunk, _, _ in split_clauses_with_offsets(original_findings):
        chunk_norm = normalize_for_span_check(chunk)

        if not contains_any_keyword(chunk, keywords):
            continue

        if polarity == "absent" and not contains_negation_cue(chunk):
            continue

        candidates.append(chunk)

    if candidates:
        # Prefer the shortest candidate that contains the topic keyword and negation cue if needed.
        candidates = sorted(candidates, key=len)
        return candidates[0].rstrip(".")

    # Second try substring from negation cue to topic keyword.
    # This handles "There is no pleural effusion or pneumothorax" -> "no pleural effusion or pneumothorax".
    if polarity == "absent":
        original_lower = original_findings.lower()

        keyword_positions: list[tuple[int, int]] = []
        for kw in keywords:
            for match in re.finditer(re.escape(kw.lower()), original_lower):
                keyword_positions.append((match.start(), match.end()))

        cue_positions: list[tuple[int, int]] = []
        for cue in NEGATION_CUES:
            for match in re.finditer(rf"\b{re.escape(cue.lower())}\b", original_lower):
                cue_positions.append((match.start(), match.end()))

        spans: list[str] = []
        for cue_start, _ in cue_positions:
            for _, kw_end in keyword_positions:
                if cue_start < kw_end:
                    candidate = original_findings[cue_start:kw_end]
                    # Avoid crossing sentence boundaries if possible.
                    if "." not in candidate and ";" not in candidate:
                        spans.append(candidate.strip(" ,"))

        if spans:
            spans = sorted(spans, key=len)
            return spans[0]

    # Third try fuzzy matching to find the closest original clause.
    # This is only a fallback and should not be overused.
    chunks = [chunk for chunk, _, _ in split_clauses_with_offsets(original_findings)]
    if chunks:
        best = max(
            chunks,
            key=lambda chunk: SequenceMatcher(
                None,
                normalize_for_span_check(source_span),
                normalize_for_span_check(chunk),
            ).ratio(),
        )
        score = SequenceMatcher(
            None,
            normalize_for_span_check(source_span),
            normalize_for_span_check(best),
        ).ratio()

        if score >= 0.55 and contains_any_keyword(best, keywords):
            return best.rstrip(".")

    return None


def validate_output(
    raw_obj: dict,
    original_findings: str,
    repair_spans: bool = True,
) -> SplitFindingsOutput:
    parsed = SplitFindingsOutput.model_validate(raw_obj)

    for item in parsed.findings:
        if not exact_or_normalized_span_found(item.source_span, original_findings):
            repaired_span = None

            if repair_spans:
                repaired_span = repair_source_span(
                    source_span=item.source_span,
                    original_findings=original_findings,
                    topic=item.topic,
                    polarity=item.polarity,
                )

            if repaired_span is not None and exact_or_normalized_span_found(
                repaired_span,
                original_findings,
            ):
                item.source_span = repaired_span
            else:
                raise ValueError(
                    f"source_span not found in original findings: {item.source_span}"
                )

        if not item.sentence.endswith("."):
            item.sentence = item.sentence.rstrip() + "."

        if len(item.sentence) > 300:
            raise ValueError(f"Sentence too long: {item.sentence}")

    return parsed