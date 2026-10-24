import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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


def make_client(config) -> LocalVLLMClient:
    return LocalVLLMClient(
        base_url=config["vllm"]["base_url"],
        api_key=config["vllm"]["api_key"],
        model=config["vllm"]["model"],
        temperature=config["vllm"].get("temperature", 0.0),
        top_p=config["vllm"].get("top_p", 1.0),
        max_tokens=config["vllm"].get("max_tokens", 2048),
        timeout_seconds=config["vllm"].get("timeout_seconds", 120),
    )


def process_one_report(
    report: dict[str, Any],
    system_prompt: str,
    config,
    repair_spans: bool,
) -> dict[str, Any]:
    """
    Process one report.

    Important:
    We create a client inside the worker to avoid thread-safety issues
    if LocalVLLMClient stores an internal session/client object.
    """
    client = make_client(config)

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

    return output_obj


def main(config) -> None:
    data_split = config["data_split"]

    input_path = (
        Path(config["data_folder"])
        / f"cailab_mimic-cxr_report_annotations_{data_split}.csv" # Original MIMIC CXR report findings
    )

    system_prompt = load_text(Path("prompts") / config["prompt_file"])

    df = load_table(
        path=input_path,
        input_format=config["input_format"],
    )

    limit = config["processing"].get("limit")
    if limit is not None:
        df = df.head(int(limit))

    # output_dir = Path("outputs") / data_split
    output_dir = Path("outputs") / "imagenome_v1" / data_split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "normalized_findings.jsonl"
    failed_path = output_dir / "failed_reports.jsonl"

    overwrite = bool(config["processing"].get("overwrite", False))
    if overwrite:
        output_path.unlink(missing_ok=True)
        failed_path.unlink(missing_ok=True)

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

    pending_reports = []
    skipped_count = 0

    for report in reports:
        key = make_key(report)

        if key in done_keys:
            skipped_count += 1
            continue

        pending_reports.append(report)

    num_workers = int(config["processing"].get("num_workers", 16))
    repair_spans = bool(config["processing"].get("repair_spans", True))

    print(f"Loaded {len(reports)} reports with non-empty findings.")
    print(f"Already completed reports skipped: {skipped_count}")
    print(f"Reports left to process: {len(pending_reports)}")
    print(f"Using num_workers={num_workers}")
    print(f"Writing outputs to: {output_path}")
    print(f"Writing failures to: {failed_path}")

    success_count = 0
    failure_count = 0

    max_in_flight = int(config["processing"].get("max_in_flight", num_workers * 4))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        pending_iter = iter(pending_reports)
        future_to_report = {}

        # Submit initial window of jobs.
        for _ in range(min(max_in_flight, len(pending_reports))):
            report = next(pending_iter)
            future = executor.submit(
                process_one_report,
                report,
                system_prompt,
                config,
                repair_spans,
            )
            future_to_report[future] = report

        pbar = tqdm(total=len(pending_reports), desc="Splitting findings")

        while future_to_report:
            for future in as_completed(future_to_report):
                report = future_to_report.pop(future)

                try:
                    output_obj = future.result()
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

                pbar.update(1)

                try:
                    next_report = next(pending_iter)
                    next_future = executor.submit(
                        process_one_report,
                        next_report,
                        system_prompt,
                        config,
                        repair_spans,
                    )
                    future_to_report[next_future] = next_report
                except StopIteration:
                    pass

                break

        pbar.close()

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