"""
Schema and vocab for the ImaGenome phrase-level pass (v2).

Unlike schemas.py (report-level, 4 fixed boxes), the JSON schema here is built
per request: the bbox enum is restricted to the phrase's own ImaGenome boxes,
so guided decoding cannot emit a box that ImaGenome did not attach.
"""

from typing import Any


Polarity = ["present", "absent", "uncertain"]

# "regional" findings go into per-(image, bbox) targets.
# "global" findings ("No acute cardiopulmonary process") go to a separate dataset.
Scope = ["regional", "global"]

# ImaGenome attribute names (attributes column, third field), minus the
# nlp|normal/abnormal and severity/laterality/temporal/texture modifiers.
# Using the same names lets us check LLM output against `attributes` directly.
ENTITIES = [
    # lungs
    "lung opacity",
    "airspace opacity",
    "atelectasis",
    "linear/patchy atelectasis",
    "lobar/segmental collapse",
    "consolidation",
    "infiltration",
    "pulmonary edema/hazy opacity",
    "vascular congestion",
    "vascular redistribution",
    "pneumonia",
    "aspiration",
    "lung lesion",
    "mass/nodule (not otherwise specified)",
    "multiple masses/nodules",
    "calcified nodule",
    "lung cancer",
    "granulomatous disease",
    "pleural/parenchymal scarring",
    "increased reticular markings/ild pattern",
    "interstitial lung disease",
    "bronchiectasis",
    "cyst/bullae",
    "hyperaeration",
    "copd/emphysema",
    "alveolar hemorrhage",
    "fluid overload/heart failure",
    "low lung volumes",
    # pleura
    "pleural effusion",
    "costophrenic angle blunting",
    "pneumothorax",
    "hydropneumothorax",
    # heart / mediastinum / hila
    "enlarged cardiac silhouette",
    "pericardial effusion",
    "mediastinal widening",
    "mediastinal displacement",
    "superior mediastinal mass/enlargement",
    "pneumomediastinum",
    "enlarged hilum",
    "tortuous aorta",
    "vascular calcification",
    "goiter",
    # diaphragm / abdomen
    "elevated hemidiaphragm",
    "diaphragmatic eventration (benign)",
    "sub-diaphragmatic air",
    "hernia",
    # bones
    "spinal fracture",
    "clavicle fracture",
    "bone lesion",
    "spinal degenerative changes",
    "scoliosis",
    "shoulder osteoarthritis",
    # tubes, lines, devices
    "endotracheal tube",
    "tracheostomy tube",
    "enteric tube",
    "chest tube",
    "pigtail catheter",
    "mediastinal drain",
    "picc",
    "ij line",
    "subclavian line",
    "swan-ganz catheter",
    "chest port",
    "intra-aortic balloon pump",
    "cardiac pacer and wires",
    "prosthetic valve",
    "cabg grafts",
    "aortic graft/repair",
    # technical
    "rotated",
    "breast/nipple shadows",
    "artifact",
    # anything not covered above (normal anatomy statements, fractures of
    # other bones, etc.)
    "other",
]

# Child -> parents, used only when comparing against `attributes`.
# ImaGenome adds the parent label alongside the child (e.g. atelectasis also
# gets "lung opacity"), while the LLM emits the most specific one.
ENTITY_PARENTS = {
    "airspace opacity": ["lung opacity"],
    "atelectasis": ["lung opacity"],
    "linear/patchy atelectasis": ["atelectasis", "lung opacity"],
    "lobar/segmental collapse": ["atelectasis", "lung opacity"],
    "consolidation": ["lung opacity"],
    "infiltration": ["lung opacity"],
    "pulmonary edema/hazy opacity": ["lung opacity"],
    "pneumonia": ["lung opacity"],
    "aspiration": ["pneumonia", "lung opacity"],
    "lung lesion": ["lung opacity"],
    "mass/nodule (not otherwise specified)": ["lung lesion", "lung opacity"],
    "multiple masses/nodules": ["lung lesion", "lung opacity"],
    "calcified nodule": ["lung lesion", "lung opacity"],
    "lung cancer": ["lung lesion", "lung opacity"],
    "pleural/parenchymal scarring": ["lung opacity"],
    "increased reticular markings/ild pattern": ["lung opacity"],
    "interstitial lung disease": ["lung opacity"],
    "pleural effusion": ["lung opacity"],
    "costophrenic angle blunting": ["pleural effusion"],
    "hydropneumothorax": ["pneumothorax", "pleural effusion"],
}


def build_response_schema(candidate_boxes: list[str]) -> dict[str, Any]:
    box_items: dict[str, Any] = {"type": "string", "enum": list(candidate_boxes)}

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["findings"],
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["finding", "entity", "polarity", "scope", "boxes"],
                    "properties": {
                        "finding": {"type": "string", "minLength": 3, "maxLength": 200},
                        "entity": {"type": "string", "enum": ENTITIES},
                        "polarity": {"type": "string", "enum": Polarity},
                        "scope": {"type": "string", "enum": Scope},
                        "boxes": {
                            "type": "array",
                            "items": box_items,
                            "uniqueItems": True,
                            "maxItems": len(candidate_boxes),
                        },
                    },
                },
            },
        },
    }


def validate_phrase_output(raw_obj: dict[str, Any], candidate_boxes: list[str]) -> list[dict[str, Any]]:
    """
    Hard checks only: structure, enums, boxes subset of candidates.
    Agreement with ImaGenome attributes is a soft check (check_phrase_outputs.py).
    """
    findings = raw_obj.get("findings")
    if not isinstance(findings, list):
        raise ValueError("Missing findings list.")

    allowed = set(candidate_boxes)
    cleaned: list[dict[str, Any]] = []

    for item in findings:
        finding = str(item["finding"]).strip()
        if len(finding) < 3:
            raise ValueError(f"Finding too short: {finding!r}")
        if not finding.endswith("."):
            finding = finding + "."
        finding = finding[0].upper() + finding[1:]

        if item["entity"] not in ENTITIES:
            raise ValueError(f"Unknown entity: {item['entity']}")
        if item["polarity"] not in Polarity:
            raise ValueError(f"Unknown polarity: {item['polarity']}")
        if item["scope"] not in Scope:
            raise ValueError(f"Unknown scope: {item['scope']}")

        boxes = list(dict.fromkeys(item["boxes"]))
        bad = set(boxes) - allowed
        if bad:
            raise ValueError(f"Boxes not in candidates: {sorted(bad)}")

        # Global findings are not grounded to a box.
        if item["scope"] == "global":
            boxes = []
        elif not boxes:
            raise ValueError(f"Regional finding with no boxes: {finding!r}")

        cleaned.append(
            {
                "finding": finding,
                "entity": item["entity"],
                "polarity": item["polarity"],
                "scope": item["scope"],
                "boxes": boxes,
            }
        )

    return cleaned
