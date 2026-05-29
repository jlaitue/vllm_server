import argparse
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf
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

import time
from rich.console import Console
from rich.panel import Panel

def print_config(config):
    console = Console()
    yaml_str = OmegaConf.to_yaml(config)
    console.print(Panel.fit(yaml_str, title="🔧 Loaded Configuration", border_style="cyan"))

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main(config) -> None:

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
                "dicom_id": report.get("dicom_id"),
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
                "dicom_id": report.get("dicom_id"),
                "original_findings": report["findings"],
                "error": repr(exc),
            }

            append_jsonl(failed_path, failed_obj)

    print(f"Done. Output written to: {output_path}")
    print(f"Failures written to: {failed_path}")


if __name__ == "__main__":
    # Command-line parser for config path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument('--override', nargs='*', help="Override config values from terminal using dot notation")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    OmegaConf.set_struct(config, False)  # Unlock config to be able to add new args

    # If there are overrides from the terminal, apply them
    if args.override:
        # Ex. --override model.name=evax dataset.batch_size=64 hypers.epochs=20
        cli_overrides = OmegaConf.from_dotlist(args.override)
        config = OmegaConf.merge(config, cli_overrides)
    
    print_config(config)
    time.sleep(1) # Just so you can quickly verify your config.

    main(config)