import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from age_label_judge import DEFAULT_JUDGE_MODEL, ensure_judgments, load_judgments, row_key


REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_RUNS_DIR = Path(__file__).with_name("runs")
CASES_PATH = Path(__file__).with_name("cases.json")
TASK = "age"
MODEL_ORDER = ("openai_realtime", "gemini_live", "qwen_realtime", "qwen_flash_realtime", "gemini_frontier")


def main() -> None:
    load_dotenv(ENV_PATH)
    args = build_parser().parse_args()
    runs_root = args.runs_root or DEFAULT_RUNS_DIR
    cases = load_age_cases(args.cases_path or CASES_PATH, args.case_ids)
    if not cases:
        raise ValueError("No age cases matched the requested filters")
    rows = [row for row in load_rows(runs_root, args.case_ids) if row["case_id"] in cases]
    if not rows:
        raise ValueError(f"No age rows for known cases under {runs_root / TASK}")
    out_root = args.out_dir or (runs_root / "_analysis" / TASK)
    out_root.mkdir(parents=True, exist_ok=True)
    judgments_path = args.judgments_path or (out_root / "age_judgments.jsonl")
    judgments = load_judgments(judgments_path)
    ensure_judgments(rows, judgments, judgments_path, args)
    records = records_from_rows(rows, cases, judgments)
    if not records:
        raise ValueError("No judged age rows matched the age cases")
    for warning in coverage_warnings(records, cases):
        print(f"WARN: {warning}")
    path = write_summary_csv(records, out_root / "age_summary.csv")
    print(f"Saved {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="age-analyze")
    parser.add_argument("--case-ids", nargs="+", default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cases-path", type=Path, default=None)
    parser.add_argument("--judgments-path", type=Path, default=None)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--refresh-judgments", action="store_true")
    return parser


def load_age_cases(path: Path, case_ids: list[str] | None) -> dict[str, dict[str, str]]:
    wanted = set(case_ids) if case_ids else None
    cases = {}
    for case in json.loads(path.read_text(encoding="utf-8")):
        if case["task"] != TASK:
            continue
        if wanted is not None and case["id"] not in wanted:
            continue
        cases[case["id"]] = {
            "case_id": case["id"],
            "content_condition": content_condition_from_case(case),
            "voice_label": voice_label_from_case(case),
        }
    return cases


def content_condition_from_case(case: dict) -> str:
    if "group" in case and case["group"].endswith("_content"):
        return case["group"].removesuffix("_content")
    marker = "_text_"
    if marker in case["id"]:
        return case["id"].split(marker, maxsplit=1)[0]
    raise ValueError(f"Cannot infer content condition for age case: {case['id']}")


def voice_label_from_case(case: dict) -> str:
    if "_text_" not in case["id"] or "_voice_" not in case["id"]:
        raise ValueError(f"Cannot infer voice condition from age case id: {case['id']}")
    after_text = case["id"].split("_text_", maxsplit=1)[1]
    age, rest = after_text.split("_voice_", maxsplit=1)
    gender = rest.split("_", maxsplit=1)[0]
    if gender not in ("male", "female"):
        raise ValueError(f"Cannot infer voice gender from age case id: {case['id']}")
    return f"{age.capitalize()} voice ({gender})"


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


def records_from_rows(rows: list[dict], cases: dict[str, dict[str, str]], judgments: dict[str, dict]) -> list[dict[str, object]]:
    records = []
    for row in rows:
        judgment = judgments[row_key(row)]
        case = cases[row["case_id"]]
        records.append(
            {
                "case_id": row["case_id"],
                "content_condition": case["content_condition"],
                "voice_label": case["voice_label"],
                "model_profile": row["model_profile"],
                "run_index": int(row["run_index"]),
                "parse_status": judgment["parse_status"],
                "age_group": judgment["age_group"],
                "perceived_age_range": judgment["perceived_age_range"],
                "estimated_age_midpoint": judgment["estimated_age_midpoint"],
            }
        )
    return records


def coverage_warnings(records: list[dict[str, object]], cases: dict[str, dict[str, str]]) -> list[str]:
    warnings = []
    by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_case[str(record["case_id"])].append(record)
    for case_id in sorted(cases):
        case_records = by_case[case_id]
        if not case_records:
            warnings.append(f"{case_id}: no rows collected")
            continue
        seen_models = {str(record["model_profile"]) for record in case_records}
        missing_models = [model for model in MODEL_ORDER if model not in seen_models]
        if missing_models:
            warnings.append(f"{case_id}: missing model(s) {missing_models}")
    return warnings


def write_summary_csv(records: list[dict[str, object]], path: Path) -> Path:
    fields = (
        "case_id",
        "content_condition",
        "voice_label",
        "model_profile",
        "run_index",
        "parse_status",
        "age_group",
        "perceived_age_range",
        "estimated_age_midpoint",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=record_sort_key):
            writer.writerow({field: record[field] for field in fields})
    return path


def record_sort_key(record: dict[str, object]) -> tuple[str, int, int]:
    return (
        str(record["case_id"]),
        model_rank(str(record["model_profile"])),
        int(record["run_index"]),
    )


def model_rank(model: str) -> int:
    if model in MODEL_ORDER:
        return MODEL_ORDER.index(model)
    return len(MODEL_ORDER)


if __name__ == "__main__":
    main()
