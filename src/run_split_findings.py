import argparse
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from io_utils import (
    append_jsonl,
    iter_reports,
    load_done_keys,
    load_table,
    make_key,
)
from llm_client import LocalVLLMClient
from validate import validate_output


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    system_prompt = load_text("prompts/split_findings_system.txt")

    df = load_table(
        path=config["input_path"],
        input_format=config["input_format"],
    )

    limit = config["processing"].get("limit")
    if limit is not None:
        df = df.head(int(limit))

    client = LocalVLLMClient(
        base_url=config["vllm"]["base_url"],
        api_key=config["vllm"]["api_key"],
        model=config["vllm"]["model"],
        temperature=config["vllm"].get("temperature", 0.0),
        top_p=config["vllm"].get("top_p", 1.0),
        max_tokens=config["vllm"].get("max_tokens", 2048),
        timeout_seconds=config["vllm"].get("timeout_seconds", 120),
    )

    output_path = config["output_path"]
    failed_path = config["failed_path"]

    done_keys = set()
    if config["processing"].get("resume", True):
        done_keys = load_done_keys(output_path)

    reports = list(
        iter_reports(
            df=df,
            findings_column=config["findings_column"],
            id_columns=config["id_columns"],
        )
    )

    print(f"Loaded {len(reports)} reports with non-empty findings.")

    for report in tqdm(reports, desc="Splitting findings"):
        key = make_key(report)

        if key in done_keys:
            continue

        try:
            raw_output = client.split_findings(
                system_prompt=system_prompt,
                findings_text=report["findings"],
            )

            parsed = validate_output(
                raw_obj=raw_output,
                original_findings=report["findings"],
            )

            output_obj: dict[str, Any] = {
                "row_index": report["row_index"],
                "subject_id": report.get("subject_id"),
                "study_id": report.get("study_id"),
                "report_id": report.get("report_id"),
                "original_findings": report["findings"],
                "findings": [item.model_dump() for item in parsed.findings],
                "model": config["vllm"]["model"],
            }

            append_jsonl(output_path, output_obj)

        except Exception as exc:
            failed_obj = {
                "row_index": report["row_index"],
                "subject_id": report.get("subject_id"),
                "study_id": report.get("study_id"),
                "report_id": report.get("report_id"),
                "original_findings": report["findings"],
                "error": repr(exc),
            }

            append_jsonl(failed_path, failed_obj)

    print(f"Done. Output written to: {output_path}")
    print(f"Failures written to: {failed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)