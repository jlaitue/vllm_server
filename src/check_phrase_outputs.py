"""
Soft checks for run_phrase_split.py outputs: agreement with ImaGenome
`attributes`, leftover comparison language, and box usage.

ImaGenome attributes are noisy, so disagreement means "look at this unit",
not "the LLM is wrong". Writes flagged units for manual review.

Usage:
  python src/check_phrase_outputs.py outputs/imagenome_v2/runs/<run_name>
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from phrase_schemas import ENTITIES, ENTITY_PARENTS


ATTR_CATEGORIES = {"anatomicalfinding", "disease", "tubesandlines", "device", "technicalassessment"}

COMPARISON_LEAK_RE = re.compile(
    r"\b(?:unchanged|stable|increas\w*|decreas\w*|improv\w*|worsen\w*|prior|previous(?:ly)?|again|"
    r"interval|new(?:ly)?|resolv\w*|persist\w*|remains?|since|compared)\b",
    re.IGNORECASE,
)


def attr_labels(attributes: list[str]) -> set[tuple[str, str]]:
    labels = set()
    for attr in attributes:
        category, flag, name = attr.split("|", 2)
        if category in ATTR_CATEGORIES and name in ENTITIES:
            labels.add((name, "yes" if flag == "yes" else "no"))
    return labels


def llm_labels(findings: list[dict], closure: bool) -> set[tuple[str, str]]:
    labels = set()
    for item in findings:
        if item["entity"] == "other":
            continue
        # ImaGenome has no "uncertain"; hedged findings are tagged yes.
        flag = "no" if item["polarity"] == "absent" else "yes"
        names = [item["entity"]]
        if closure:
            names += ENTITY_PARENTS.get(item["entity"], [])
        labels.update((name, flag) for name in names)
    return labels


def check_unit(obj: dict) -> dict:
    attrs = attr_labels(obj["attributes"])
    llm_closed = llm_labels(obj["findings"], closure=True)
    llm_specific = llm_labels(obj["findings"], closure=False)

    missed = attrs - llm_closed
    extra = {label for label in llm_specific if label not in attrs}
    flipped = {
        name for name, flag in missed if (name, "no" if flag == "yes" else "yes") in llm_closed
    }
    leaks = [f["finding"] for f in obj["findings"] if COMPARISON_LEAK_RE.search(f["finding"])]

    return {
        "n_attr": len(attrs),
        "n_matched": len(attrs) - len(missed),
        "missed": sorted(f"{n}|{f}" for n, f in missed),
        "extra": sorted(f"{n}|{f}" for n, f in extra),
        "flipped": sorted(flipped),
        "comparison_leaks": leaks,
    }


def main(args) -> None:
    run_dir = Path(args.run_dir)
    outputs = [json.loads(l) for l in open(run_dir / "phrase_outputs.jsonl", encoding="utf-8")]
    failed_path = run_dir / "failed_units.jsonl"
    n_failed = sum(1 for _ in open(failed_path)) if failed_path.exists() else 0

    totals = Counter()
    entity_counts = Counter()
    flagged = []

    for obj in outputs:
        result = check_unit(obj)
        findings = obj["findings"]

        totals["units"] += 1
        totals["findings"] += len(findings)
        totals["empty"] += not findings
        totals["retried"] += obj.get("n_attempts", 1) > 1
        totals["attr_total"] += result["n_attr"]
        totals["attr_matched"] += result["n_matched"]
        totals["units_with_missed"] += bool(result["missed"])
        totals["units_with_extra"] += bool(result["extra"])
        totals["units_with_flip"] += bool(result["flipped"])
        totals["units_with_leak"] += bool(result["comparison_leaks"])
        for f in findings:
            totals[f"polarity_{f['polarity']}"] += 1
            totals[f"scope_{f['scope']}"] += 1
            entity_counts[f["entity"]] += 1
            totals["boxes_used"] += len(f["boxes"])
            totals["boxes_offered"] += len(obj["candidate_boxes"]) if f["scope"] == "regional" else 0

        if result["missed"] or result["flipped"] or result["comparison_leaks"]:
            flagged.append({**obj, "check": result})

    n = max(totals["units"], 1)
    print(f"Run: {run_dir}")
    print(f"Units ok: {totals['units']:,}   failed: {n_failed:,}   ({n_failed / (n + n_failed):.2%} fail)")
    print(f"Retried (first attempt invalid): {totals['retried'] / n:.2%}")
    print(f"Findings/unit: {totals['findings'] / n:.2f}   empty units: {totals['empty'] / n:.2%}")
    print(
        "Polarity: "
        + ", ".join(f"{p}={totals[f'polarity_{p}']:,}" for p in ("present", "absent", "uncertain"))
    )
    print(f"Scope: regional={totals['scope_regional']:,}, global={totals['scope_global']:,}")
    print(f"Boxes used / offered (regional): {totals['boxes_used'] / max(totals['boxes_offered'], 1):.2%}")
    print(f"Attribute recall (label+polarity): {totals['attr_matched'] / max(totals['attr_total'], 1):.2%}")
    print(f"Units with missed attr: {totals['units_with_missed'] / n:.2%}")
    print(f"Units with extra entity: {totals['units_with_extra'] / n:.2%}")
    print(f"Units with polarity flip: {totals['units_with_flip'] / n:.2%}")
    print(f"Units with comparison leak: {totals['units_with_leak'] / n:.2%}")
    print(f"Entity 'other': {entity_counts['other'] / max(totals['findings'], 1):.2%} of findings")
    print("Top entities:", entity_counts.most_common(12))

    flagged_path = run_dir / "flagged_units.jsonl"
    with open(flagged_path, "w", encoding="utf-8") as f:
        for obj in flagged:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(flagged):,} flagged units to {flagged_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    main(parser.parse_args())
