"""
Step 2 of the ImaGenome v2 pipeline: one LLM call per (phrase, box set) unit.

Reads units from build_phrase_units.py (units.jsonl or pilot_units.jsonl),
writes one JSON line per unit. Resumable by unit_id.
"""

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tqdm import tqdm

from io_utils import append_jsonl
from llm_client import LocalVLLMClient
from phrase_schemas import build_response_schema, validate_phrase_output
from run_split_findings_concurrent import load_text, make_client, print_config


def load_units(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_unit_ids(*paths: Path) -> set[str]:
    done = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["unit_id"])
    return done


def make_user_prompt(unit: dict[str, Any]) -> str:
    return (
        f"Sentence: {unit['phrase']}\n"
        f"Candidate boxes: {json.dumps(unit['candidate_boxes'])}"
    )


def process_one_unit(
    unit: dict[str, Any],
    client: LocalVLLMClient,
    system_prompt: str,
    retry_temperature: float,
) -> dict[str, Any]:
    schema = build_response_schema(unit["candidate_boxes"])
    user_prompt = make_user_prompt(unit)

    attempts = []
    # Greedy first; on a hard validation failure resample once.
    for temperature in (None, retry_temperature):
        raw = client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            schema_name="phrase_findings",
            temperature=temperature,
        )
        try:
            findings = validate_phrase_output(raw, unit["candidate_boxes"])
            return {
                "unit_id": unit["unit_id"],
                "phrase": unit["phrase"],
                "candidate_boxes": unit["candidate_boxes"],
                "attributes": unit["attributes"],
                "findings": findings,
                "n_attempts": len(attempts) + 1,
            }
        except (KeyError, TypeError, ValueError) as exc:
            attempts.append({"raw": raw, "error": repr(exc)})

    raise ValueError(json.dumps(attempts, ensure_ascii=False))


def main(config) -> None:
    units_path = Path(config["units_file"])
    out_dir = Path(config["out_dir"]) / config["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / "phrase_outputs.jsonl"
    failed_path = out_dir / "failed_units.jsonl"

    if config["processing"].get("overwrite", False):
        output_path.unlink(missing_ok=True)
        failed_path.unlink(missing_ok=True)

    # Save the exact config next to the outputs so runs are comparable.
    OmegaConf.save(config, out_dir / "config.yaml")

    system_prompt = load_text(Path("prompts") / config["prompt_file"])
    units = load_units(units_path)

    limit = config["processing"].get("limit")
    if limit is not None:
        units = units[: int(limit)]

    done = set()
    if config["processing"].get("resume", True):
        # Failed units are retried on resume; only successes are skipped.
        done = load_done_unit_ids(output_path)
        failed_path.unlink(missing_ok=True)

    pending = [u for u in units if u["unit_id"] not in done]

    print(f"Loaded {len(units):,} units from {units_path}")
    print(f"Already done: {len(units) - len(pending):,}; to process: {len(pending):,}")
    print(f"Writing outputs to: {output_path}")

    client = make_client(config)
    retry_temperature = float(config["processing"].get("retry_temperature", 0.7))
    num_workers = int(config["processing"].get("num_workers", 64))
    max_in_flight = int(config["processing"].get("max_in_flight", num_workers * 2))

    success_count = 0
    failure_count = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        pending_iter = iter(pending)
        in_flight = {}

        def submit_next() -> None:
            unit = next(pending_iter, None)
            if unit is not None:
                future = executor.submit(process_one_unit, unit, client, system_prompt, retry_temperature)
                in_flight[future] = unit

        for _ in range(max_in_flight):
            submit_next()

        pbar = tqdm(total=len(pending), desc="Splitting phrases")

        while in_flight:
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                unit = in_flight.pop(future)
                try:
                    output_obj = future.result()
                    output_obj["model"] = config["vllm"]["model"]
                    output_obj["prompt_file"] = config["prompt_file"]
                    append_jsonl(output_path, output_obj)
                    success_count += 1
                except Exception as exc:
                    append_jsonl(
                        failed_path,
                        {
                            "unit_id": unit["unit_id"],
                            "phrase": unit["phrase"],
                            "candidate_boxes": unit["candidate_boxes"],
                            "error": repr(exc)[:4000],
                        },
                    )
                    failure_count += 1

                pbar.update(1)
                submit_next()

        pbar.close()

    elapsed = time.time() - start
    print("Done.")
    print(f"Successful units: {success_count:,}")
    print(f"Failed units: {failure_count:,}")
    print(f"Elapsed: {elapsed / 60:.1f} min ({success_count / max(elapsed, 1e-6):.1f} units/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_phrase.yaml")
    parser.add_argument("--override", nargs="*", help="Override config values using dot notation")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    OmegaConf.set_struct(config, False)

    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))

    print_config(config)
    main(config)
