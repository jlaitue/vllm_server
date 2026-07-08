import argparse
import time
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
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


def print_config(config) -> None:
    console = Console()
    yaml_str = OmegaConf.to_yaml(config)
    console.print(Panel.fit(yaml_str, title="Loaded Configuration", border_style="cyan"))


def load_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(config) -> None:
    data_split = config["data_split"]

    input_path = Path(config["data_folder"]) / f"cailab_mimic-cxr_report_annotations_{data_split}.csv"
    prompt_path = Path("prompts") / config["prompt_file"]
    system_prompt = load_text(prompt_path)

    df = load_table(
        path=input_path,
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

    output_dir = Path("outputs") / data_split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "normalized_findings.jsonl"
    failed_path = output_dir / "failed_reports.jsonl"

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
    print(f"Writing outputs to: {output_path}")
    print(f"Writing failures to: {failed_path}")

    success_count = 0
    failure_count = 0
    skipped_count = 0

    repair_spans = config["processing"].get("repair_spans", True)

    for report in tqdm(reports, desc="Splitting findings"):
        key = make_key(report)

        if key in done_keys:
            skipped_count += 1
            continue

        try:
            raw_output = client.split_findings(
                system_prompt=system_prompt,
                findings_text=report["findings"],
            )

            parsed = validate_output(
                raw_obj=raw_output,
                original_findings=report["findings"],
                repair_spans=repair_spans,
            )

            output_obj: dict[str, Any] = {
                "row_index": report["row_index"],
                "subject_id": report.get("subject_id"),
                "study_id": report.get("study_id"),
                "dicom_id": report.get("dicom_id"),
                "original_findings": report["findings"],
                "findings": [item.model_dump() for item in parsed.findings],
                "model": config["vllm"]["model"],
                "prompt_file": config["prompt_file"],
            }

            append_jsonl(output_path, output_obj)
            success_count += 1

        except Exception as exc:
            failed_obj = {
                "row_index": report["row_index"],
                "subject_id": report.get("subject_id"),
                "study_id": report.get("study_id"),
                "dicom_id": report.get("dicom_id"),
                "original_findings": report["findings"],
                "error": repr(exc),
                "model": config["vllm"]["model"],
                "prompt_file": config["prompt_file"],
            }

            append_jsonl(failed_path, failed_obj)
            failure_count += 1

    print("Done.")
    print(f"Successful reports: {success_count}")
    print(f"Failed reports: {failure_count}")
    print(f"Skipped reports: {skipped_count}")
    print(f"Output written to: {output_path}")
    print(f"Failures written to: {failed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        help="Override config values from terminal using dot notation",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    OmegaConf.set_struct(config, False)

    if args.override:
        cli_overrides = OmegaConf.from_dotlist(args.override)
        config = OmegaConf.merge(config, cli_overrides)

    print_config(config)

    startup_sleep_seconds = config.get("startup_sleep_seconds", 0)
    if startup_sleep_seconds:
        time.sleep(float(startup_sleep_seconds))

    main(config)