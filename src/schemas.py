from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Region = Literal[
    "lungs",
    "pleura",
    "heart_cardiomediastinum",
    "mediastinum_hila",
    "bones_chest_wall",
    "diaphragm_subdiaphragm",
    "support_devices",
    "global_or_multiregion",
]


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

BBoxName = Literal[
    "right_lung",
    "left_lung",
    "mediastinum",
    "cardiac_silhouette",
]


class FindingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence: str = Field(min_length=3)
    region: Region
    bbox_names: list[BBoxName] = Field(max_length=4)
    topic: Topic
    polarity: Polarity
    source_span: str = Field(min_length=1)

    @field_validator("bbox_names")
    @classmethod
    def validate_unique_boxes(cls, names: list[BBoxName]) -> list[BBoxName]:
        if len(names) != len(set(names)):
            raise ValueError("bbox_names must contain unique names")
        return names


class SplitFindingsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[FindingItem]


REPORT_SCHEMA = SplitFindingsOutput.model_json_schema()