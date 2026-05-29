from typing import Literal

from pydantic import BaseModel, Field


Topic = Literal[
    "disease-atelectasis",
    "disease-cardiomegaly",
    "disease-consolidation",
    "disease-edema",
    "disease-enlarged cardiomediastinum",
    "disease-fracture",
    "disease-lung lesion",
    "disease-lung opacity",
    "disease-pleural effusion",
    "disease-pleural other",
    "disease-pneumonia",
    "disease-pneumothorax",
    "organ-heart",
    "organ-lungs",
    "organ-pleura",
    "organ-mediastinum",
    "organ-bones",
    "organ-diaphragm",
    "support devices",
    "patient status",
    "other",
]

Polarity = Literal["present", "absent", "uncertain"]


class FindingItem(BaseModel):
    topic: Topic
    sentence: str = Field(min_length=3)
    polarity: Polarity
    source_span: str = Field(min_length=1)


class SplitFindingsOutput(BaseModel):
    findings: list[FindingItem]


REPORT_SCHEMA = SplitFindingsOutput.model_json_schema()