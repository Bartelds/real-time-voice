import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from accent_label_judge import (
    DEFAULT_JUDGE_MODEL,
    ensure_judgments,
    ensure_standardization,
    extracted_labels,
    load_judgments,
    load_standardization,
    row_key,
)


REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_RUNS_DIR = Path(__file__).with_name("runs")
CASES_PATH = Path(__file__).with_name("cases.json")
TASK = "accent_perception"
MODEL_ORDER = ("openai_realtime", "gemini_live", "qwen_realtime", "qwen_flash_realtime", "gemini_frontier")


def main() -> None:
    load_dotenv(ENV_PATH)
    args = build_parser().parse_args()
    runs_root = args.runs_root or DEFAULT_RUNS_DIR
    cases = load_accent_cases(args.cases_path or CASES_PATH, args.case_ids, args.content_condition)
    if not cases:
        raise ValueError("No accent_perception cases matched the requested filters")
    rows = [row for row in load_rows(runs_root, args.case_ids) if row["case_id"] in cases]
    if not rows:
        raise ValueError(f"No accent_perception rows for known cases under {runs_root / TASK}")
    cache_root = runs_root / "_analysis" / TASK
    out_root = args.out_dir or filtered_out_root(cache_root, args.content_condition)
    cache_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    judgments_path = args.judgments_path or (cache_root / "accent_label_judgments.jsonl")
    standardization_path = args.standardization_path or (cache_root / "accent_label_standardization.json")
    judgments = load_judgments(judgments_path)
    ensure_judgments(rows, judgments, judgments_path, args)
    labels = extracted_labels(rows, set(cases), judgments)
    standardization = load_standardization(standardization_path)
    ensure_standardization(labels, standardization, standardization_path, args)
    summaries = summarize(rows, cases, judgments, standardization)
    if not summaries:
        raise ValueError("No judged result rows matched the accent_perception cases")
    for warning in coverage_warnings(summaries, cases):
        print(f"WARN: {warning}")
    path = write_summary_csv(summaries, out_root / "accent_summary.csv")
    print(f"Saved {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="accent-analyze")
    parser.add_argument("--case-ids", nargs="+", default=None)
    parser.add_argument("--content-condition", choices=("italy", "japan", "netherlands"), default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cases-path", type=Path, default=None)
    parser.add_argument("--judgments-path", type=Path, default=None)
    parser.add_argument("--standardization-path", type=Path, default=None)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--refresh-judgments", action="store_true")
    parser.add_argument("--refresh-standardization", action="store_true")
    return parser


def filtered_out_root(cache_root: Path, content_condition: str | None) -> Path:
    if content_condition is None:
        return cache_root
    return cache_root / content_condition


def load_accent_cases(path: Path, case_ids: list[str] | None, content_condition: str | None) -> dict[str, dict[str, str]]:
    wanted = set(case_ids) if case_ids else None
    cases = {}
    for case in json.loads(path.read_text(encoding="utf-8")):
        if case["task"] != TASK:
            continue
        if wanted is not None and case["id"] not in wanted:
            continue
        case_content_condition = content_condition_from_case(case)
        if content_condition is not None and case_content_condition != content_condition:
            continue
        cases[case["id"]] = {
            "case_id": case["id"],
            "actual_accent_folder": accent_folder_from_audio(case["audio"]),
            "voice_label": voice_label_from_folder(accent_folder_from_audio(case["audio"])),
            "content_condition": case_content_condition,
        }
    return cases


def accent_folder_from_audio(audio: str) -> str:
    parts = Path(audio).parts
    if "accents" not in parts:
        raise ValueError(f"Accent case audio path does not include accents folder: {audio}")
    index = parts.index("accents")
    if len(parts) <= index + 1:
        raise ValueError(f"Accent case audio path does not include accent folder: {audio}")
    return parts[index + 1]


def content_condition_from_case(case: dict) -> str:
    if "group" in case and case["group"].endswith("_content"):
        return case["group"].removesuffix("_content")
    marker = "_text_"
    if marker in case["id"]:
        return case["id"].split(marker, maxsplit=1)[0]
    raise ValueError(f"Cannot infer content condition for accent case: {case['id']}")


def voice_label_from_folder(folder: str) -> str:
    parts = folder.split("_")
    if len(parts) < 4 or parts[1] != "accent":
        raise ValueError(f"Cannot infer voice label from accent folder: {folder}")
    region = parts[0]
    gender = parts[2]
    if gender not in ("male", "female"):
        raise ValueError(f"Cannot infer voice gender from accent folder: {folder}")
    return f"{region.capitalize()}-accented English ({gender})"


def load_rows(runs_root: Path, case_ids: list[str] | None) -> list[dict]:
    task_dir = runs_root / TASK
    if not task_dir.exists():
        return []
    rows = []
    wanted = set(case_ids) if case_ids else None
    for results_path in sorted(task_dir.rglob("results.jsonl")):
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["task"] != TASK:
                continue
            if wanted is not None and row["case_id"] not in wanted:
                continue
            rows.append(row)
    return rows


def summarize(rows: list[dict], cases: dict[str, dict[str, str]], judgments: dict[str, dict], standardization: dict[str, str]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["case_id"], row["model_profile"])].append(row)
    summaries = []
    for case_id, model in sorted(grouped):
        extracted = []
        display = []
        errors = 0
        missing = 0
        for row in grouped[(case_id, model)]:
            judgment = judgments[row_key(row)]
            status = judgment["parse_status"]
            if status == "error":
                errors += 1
                continue
            label = judgment["perceived_accent_label"].strip()
            if status == "missing" or not label:
                missing += 1
                continue
            extracted.append(label)
            display.append(standardization[label])
        display_counts = Counter(display).most_common()
        extracted_counts = Counter(extracted).most_common()
        top_label, top_count = display_counts[0] if display_counts else ("", 0)
        case = cases[case_id]
        summaries.append(
            {
                "case_id": case_id,
                "actual_accent_folder": case["actual_accent_folder"],
                "voice_label": case["voice_label"],
                "content_condition": case["content_condition"],
                "model_profile": model,
                "total_n": len(grouped[(case_id, model)]),
                "parsed_n": len(extracted),
                "missing_n": missing,
                "error_n": errors,
                "top_display_label": top_label,
                "top_display_count": int(top_count),
                "display_counts": display_counts,
                "extracted_counts": extracted_counts,
            }
        )
    return summaries


def coverage_warnings(summaries: list[dict[str, object]], cases: dict[str, dict[str, str]]) -> list[str]:
    warnings = []
    by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for summary in summaries:
        by_case[str(summary["case_id"])].append(summary)
    for case_id in sorted(cases):
        case_summaries = by_case[case_id]
        if not case_summaries:
            warnings.append(f"{case_id}: no rows collected")
            continue
        seen_models = {str(summary["model_profile"]) for summary in case_summaries}
        missing_models = [model for model in MODEL_ORDER if model not in seen_models]
        if missing_models:
            warnings.append(f"{case_id}: missing model(s) {missing_models}")
        totals = {int(summary["total_n"]) for summary in case_summaries}
        if len(totals) > 1:
            warnings.append(f"{case_id}: uneven runs across models {sorted(totals)}")
    return warnings


def format_counter(counts: list[tuple[str, int]]) -> str:
    return " | ".join(f"{label}: {count}" for label, count in counts)


def write_summary_csv(summaries: list[dict[str, object]], path: Path) -> Path:
    fieldnames = (
        "case_id",
        "voice_label",
        "content_condition",
        "model_profile",
        "total_n",
        "parsed_n",
        "missing_n",
        "error_n",
        "top_display_label",
        "top_display_count",
        "display_label_counts",
        "extracted_label_counts",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in sorted(summaries, key=summary_sort_key):
            row = {field: summary[field] for field in fieldnames if field in summary}
            row["display_label_counts"] = format_counter(summary["display_counts"])
            row["extracted_label_counts"] = format_counter(summary["extracted_counts"])
            writer.writerow(row)
    return path


def summary_sort_key(summary: dict[str, object]) -> tuple[str, str, str, int]:
    return (
        str(summary["actual_accent_folder"]),
        str(summary["content_condition"]),
        str(summary["case_id"]),
        model_rank(str(summary["model_profile"])),
    )


def model_rank(model: str) -> int:
    if model in MODEL_ORDER:
        return MODEL_ORDER.index(model)
    return len(MODEL_ORDER)


if __name__ == "__main__":
    main()
