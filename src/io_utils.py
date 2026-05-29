import json
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


def load_table(path: str, input_format: str) -> pd.DataFrame:
    path_obj = Path(path)

    if input_format == "csv":
        return pd.read_csv(path_obj)

    if input_format == "jsonl":
        return pd.read_json(path_obj, lines=True)

    if input_format == "parquet":
        return pd.read_parquet(path_obj)

    raise ValueError(f"Unsupported input_format: {input_format}")


def iter_reports(
    df: pd.DataFrame,
    findings_column: str,
    id_columns: dict[str, str | None],
) -> Iterator[dict[str, Any]]:
    for idx, row in df.iterrows():
        findings = row.get(findings_column)

        if not isinstance(findings, str) or not findings.strip():
            continue

        item = {
            "row_index": int(idx),
            "findings": findings.strip(),
        }

        for logical_name, col_name in id_columns.items():
            if col_name is None:
                item[logical_name] = None
            else:
                value = row.get(col_name)
                item[logical_name] = None if pd.isna(value) else value

        yield item


def append_jsonl(path: str, obj: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def make_key(obj: dict[str, Any]) -> str:
    for key in ["report_id", "study_id", "row_index"]:
        value = obj.get(key)
        if value is not None:
            return f"{key}:{value}"

    raise ValueError("Could not make key; missing report_id, study_id, and row_index.")


def load_done_keys(output_path: str) -> set[str]:
    path = Path(output_path)

    if not path.exists():
        return set()

    done = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            done.add(make_key(obj))

    return done