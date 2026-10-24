"""
Step 1 of the ImaGenome v2 pipeline: collapse the 3.5M-row region-phrase CSV
into unique LLM work units.

A unit is (normalized phrase, sorted candidate box set). ImaGenome's box
assignment is almost a deterministic function of the phrase, so this is about
the same count as unique phrases (~509k).

Outputs (outputs/imagenome_v2/):
  units.jsonl          one line per unit: unit_id, phrase, candidate_boxes, attributes, counts
  instances.csv.gz     one row per (dicom_id, phrase_id): join key back to images
  pilot_units.jsonl    stratified sample for prompt/model comparison
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SILVER_DIR = Path(
    "/scratch/ljl5178/Datasets/RADIOLOGY_REPORT/CHEST_IMAGENOME/"
    "chest-imagenome-dataset-1.0.0/silver_dataset"
)

# Report section headers that ImaGenome leaves glued to the first sentence.
HEADER_RE = re.compile(
    r"^\s*(?:FINDINGS?|IMPRESSIONS?|CONCLUSIONS?|COMMENTS?|FINAL REPORT|REPORT|NOTE)\s*:\s*",
    re.IGNORECASE,
)

COMPARISON_RE = re.compile(
    r"\b(?:unchanged|stable|increas|decreas|improv|worsen|prior|previous|again|"
    r"interval|new(?:ly)?|resolv|persist|remain|since|compar|redemonstrat)",
    re.IGNORECASE,
)


def normalize_phrase(text: str) -> str:
    text = HEADER_RE.sub("", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_unit_id(phrase: str, boxes: tuple[str, ...]) -> str:
    key = phrase + "||" + "|".join(boxes)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def stratified_pilot(units: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Half frequency-weighted (what the training data actually looks like),
    half uniform over units with comparison language or >=3 attributes
    (where splitting and temporal rewriting are hardest).
    """
    n_freq = n // 2
    # Weighted sampling without replacement (Efraimidis-Spirakis); pandas
    # refuses when the weights are this skewed.
    rng = np.random.default_rng(seed)
    keys = rng.random(len(units)) ** (1.0 / units["n_instances"].to_numpy())
    by_freq = units.iloc[np.argsort(-keys)[:n_freq]]

    rest = units.drop(by_freq.index)
    hard = rest[rest["has_comparison"] | (rest["n_attributes"] >= 3)]
    by_hard = hard.sample(n=min(n - n_freq, len(hard)), random_state=seed)

    pilot = pd.concat([by_freq.assign(pilot_stratum="freq"), by_hard.assign(pilot_stratum="hard")])
    return pilot.sample(frac=1.0, random_state=seed)


def main(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        SILVER_DIR / "chest_imagenome_region_phrase_dataset.csv",
        usecols=["subject_id", "study_id", "dicom_id", "split", "bbox_name", "phrase", "phrase_id", "attributes", "section"],
    )
    print(f"Loaded {len(df):,} rows")

    df = df[df["section"] == "finalreport"]
    print(f"finalreport rows: {len(df):,}")

    # Gold-set images; ImaGenome asks that these be excluded from silver training/eval.
    avoid = set(pd.read_csv(SILVER_DIR / "splits" / "images_to_avoid.csv")["dicom_id"])
    n_before = df["dicom_id"].nunique()
    df = df[~df["dicom_id"].isin(avoid)]
    print(f"Dropped {n_before - df['dicom_id'].nunique():,} images in images_to_avoid.csv")

    df = df.dropna(subset=["phrase"])

    instances = (
        df.groupby(["dicom_id", "phrase_id"], sort=False)
        .agg(
            subject_id=("subject_id", "first"),
            study_id=("study_id", "first"),
            split=("split", "first"),
            phrase=("phrase", "first"),
            attributes=("attributes", "first"),
            boxes=("bbox_name", lambda s: tuple(sorted(set(s)))),
        )
        .reset_index()
    )
    print(f"Phrase instances (dicom_id, phrase_id): {len(instances):,}")

    instances["phrase_norm"] = instances["phrase"].map(normalize_phrase)
    instances = instances[instances["phrase_norm"].str.len() > 0]
    instances["unit_id"] = [
        make_unit_id(p, b) for p, b in zip(instances["phrase_norm"], instances["boxes"])
    ]

    # Attributes are derived from the phrase, so they agree across instances
    # of a unit; take the first.
    units = (
        instances.groupby("unit_id", sort=False)
        .agg(
            phrase=("phrase_norm", "first"),
            boxes=("boxes", "first"),
            attributes=("attributes", "first"),
            n_instances=("dicom_id", "size"),
        )
        .reset_index()
    )
    split_counts = instances.groupby(["unit_id", "split"]).size().unstack(fill_value=0)
    units["split_counts"] = [
        {k: int(v) for k, v in row.items() if v} for row in split_counts.loc[units["unit_id"]].to_dict("records")
    ]
    units["candidate_boxes"] = units.pop("boxes").map(list)
    units["attributes"] = units["attributes"].map(lambda a: json.loads(a) if isinstance(a, str) else [])
    units["n_attributes"] = units["attributes"].map(lambda a: sum(1 for x in a if not x.startswith("nlp|")))
    units["has_comparison"] = units["phrase"].str.contains(COMPARISON_RE)
    units = units.sort_values("n_instances", ascending=False).reset_index(drop=True)
    print(f"Units (phrase, box set): {len(units):,}")
    print(f"  with comparison language: {units['has_comparison'].mean():.1%}")
    print(f"  candidate boxes per unit: {units['candidate_boxes'].map(len).describe().round(2).to_dict()}")

    units.to_json(out_dir / "units.jsonl", orient="records", lines=True, force_ascii=False)

    instances[["dicom_id", "phrase_id", "subject_id", "study_id", "split", "unit_id"]].to_csv(
        out_dir / "instances.csv.gz", index=False
    )

    pilot = stratified_pilot(units, n=args.pilot_size, seed=args.seed)
    pilot.to_json(out_dir / "pilot_units.jsonl", orient="records", lines=True, force_ascii=False)

    print(f"Wrote {out_dir / 'units.jsonl'}")
    print(f"Wrote {out_dir / 'instances.csv.gz'}")
    print(f"Wrote {out_dir / 'pilot_units.jsonl'} ({len(pilot)} units)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/imagenome_v2")
    parser.add_argument("--pilot_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
