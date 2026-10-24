"""
Step 3 of the ImaGenome v2 pipeline: join cleaned phrase units back to images
and build the radvr phase-2 SFT tables.

Outputs (in <run_dir>/dataset/):
  regions_{split}.jsonl  one row per (dicom_id, bbox_name) with 518-space box,
                         canonical-order target, and atomic findings + polarity
  global_{split}.jsonl   one row per dicom_id with scope="global" findings
                         (kept out of region targets, for later evaluation)

Negatives are kept as-is; balancing is a training-time decision in radvr.

Usage:
  python src/aggregate_regions.py outputs/imagenome_v2/runs/<run_name>
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


SILVER_DIR = Path(
    "/scratch/ljl5178/Datasets/RADIOLOGY_REPORT/CHEST_IMAGENOME/"
    "chest-imagenome-dataset-1.0.0/silver_dataset"
)

POLARITY_ORDER = {"present": 0, "uncertain": 1, "absent": 2}

# Child -> parent among ImaGenome boxes. With --propagate_to_parents a parent's
# findings are the union of its own and its children's.
BOX_PARENT = {
    "right upper lung zone": "right lung",
    "right mid lung zone": "right lung",
    "right lower lung zone": "right lung",
    "right apical zone": "right lung",
    "right costophrenic angle": "right lung",
    "left upper lung zone": "left lung",
    "left mid lung zone": "left lung",
    "left lower lung zone": "left lung",
    "left apical zone": "left lung",
    "left costophrenic angle": "left lung",
}


def canonical_sort(findings: list[dict]) -> list[dict]:
    """Positives, then uncertain, then negatives; alphabetical within each."""
    return sorted(findings, key=lambda f: (POLARITY_ORDER[f["polarity"]], f["finding"].lower()))


def dedupe(findings: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for f in findings:
        key = (f["finding"].lower(), f["polarity"])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def has_conflict(findings: list[dict]) -> bool:
    """Same entity both present and absent in one region (e.g. from two report sentences)."""
    by_entity = defaultdict(set)
    for f in findings:
        if f["entity"] != "other":
            by_entity[f["entity"]].add(f["polarity"])
    return any({"present", "absent"} <= pols for pols in by_entity.values())


def load_boxes_518() -> dict[tuple[str, str], tuple]:
    cols = [
        "dicom_id", "bbox_name", "ViewPosition",
        "x1_518", "y1_518", "x2_518", "y2_518", "visible_in_518_crop",
    ]
    boxes = pd.read_csv(SILVER_DIR / "bounding_boxes_518.csv", usecols=cols)
    boxes = boxes.drop_duplicates(["dicom_id", "bbox_name"])
    return {
        (row[0], row[1]): row[2:]
        for row in boxes[cols].itertuples(index=False, name=None)
    }


def main(args) -> None:
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    unit_findings = {}
    with open(run_dir / "phrase_outputs.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            unit_findings[obj["unit_id"]] = obj["findings"]
    print(f"Loaded {len(unit_findings):,} cleaned units")

    instances = pd.read_csv(Path(args.units_dir) / "instances.csv.gz")
    instances["done"] = instances["unit_id"].isin(unit_findings)
    print(f"Instances covered: {instances['done'].mean():.2%} of {len(instances):,}")

    # Images with any un-cleaned phrase would get silently incomplete targets.
    missing_per_image = (~instances["done"]).groupby(instances["dicom_id"]).sum()

    regions = defaultdict(list)          # (dicom_id, bbox) -> findings
    global_findings = defaultdict(list)  # dicom_id -> findings
    image_meta = {}

    for row in instances[instances["done"]].itertuples(index=False):
        image_meta[row.dicom_id] = (row.subject_id, row.study_id, row.split)
        for f in unit_findings[row.unit_id]:
            item = {"finding": f["finding"], "entity": f["entity"], "polarity": f["polarity"]}
            if f["scope"] == "global":
                global_findings[row.dicom_id].append(item)
                continue
            boxes = set(f["boxes"])
            if args.propagate_to_parents:
                boxes |= {BOX_PARENT[b] for b in boxes if b in BOX_PARENT}
            for box in boxes:
                regions[(row.dicom_id, box)].append(item)

    print(f"Region rows: {len(regions):,}; images with global findings: {len(global_findings):,}")

    boxes_518 = load_boxes_518()

    region_files = {}
    n_no_coords = 0
    for (dicom_id, bbox_name), findings in regions.items():
        key = (dicom_id, bbox_name)
        if key not in boxes_518:
            # Parent box added by propagation that ImaGenome did not list for this image.
            n_no_coords += 1
            continue
        view_position, x1, y1, x2, y2, visible = boxes_518[key]
        subject_id, study_id, split = image_meta[dicom_id]
        findings = canonical_sort(dedupe(findings))

        record = {
            "dicom_id": dicom_id,
            "subject_id": int(subject_id),
            "study_id": int(study_id),
            "split": split,
            "view_position": view_position,
            "bbox_name": bbox_name,
            "bbox_518": [float(x1), float(y1), float(x2), float(y2)],
            "visible_in_518_crop": bool(visible),
            "target": " ".join(f["finding"] for f in findings),
            "findings": findings,
            "n_present": sum(f["polarity"] == "present" for f in findings),
            "n_uncertain": sum(f["polarity"] == "uncertain" for f in findings),
            "n_absent": sum(f["polarity"] == "absent" for f in findings),
            "has_conflict": has_conflict(findings),
            "n_units_missing_in_image": int(missing_per_image.get(dicom_id, 0)),
        }

        if split not in region_files:
            region_files[split] = open(out_dir / f"regions_{split}.jsonl", "w", encoding="utf-8")
        region_files[split].write(json.dumps(record, ensure_ascii=False) + "\n")

    for fh in region_files.values():
        fh.close()
    print(f"Skipped {n_no_coords:,} propagated parent boxes with no coordinates for that image")

    global_files = {}
    for dicom_id, findings in global_findings.items():
        subject_id, study_id, split = image_meta[dicom_id]
        record = {
            "dicom_id": dicom_id,
            "subject_id": int(subject_id),
            "study_id": int(study_id),
            "split": split,
            "findings": canonical_sort(dedupe(findings)),
        }
        if split not in global_files:
            global_files[split] = open(out_dir / f"global_{split}.jsonl", "w", encoding="utf-8")
        global_files[split].write(json.dumps(record, ensure_ascii=False) + "\n")

    for fh in global_files.values():
        fh.close()
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--units_dir", default="outputs/imagenome_v2")
    parser.add_argument(
        "--propagate_to_parents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add lung-zone findings to the containing lung box",
    )
    main(parser.parse_args())
