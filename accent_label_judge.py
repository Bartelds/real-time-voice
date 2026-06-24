import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

from openai import OpenAI


DEFAULT_JUDGE_MODEL = "gpt-5.5"
PROXY_ENV_NAMES = (
    "ALL_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "SOCKS5_PROXY",
    "SOCKS_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "socks5_proxy",
    "socks_proxy",
)
JUDGE_ATTEMPTS = 4
JUDGE_SLEEP_SECONDS = 1.5


def load_judgments(path: Path) -> dict[str, dict]:
    return load_jsonl_by_key(path, "row_key")


def ensure_judgments(rows: list[dict], judgments: dict[str, dict], path: Path, args) -> None:
    missing = [row for row in rows if args.refresh_judgments or row_key(row) not in judgments]
    if not missing:
        print(f"Judge extraction: all {len(rows)} rows already cached")
        return
    print(f"Judge extraction: {len(rows) - len(missing)}/{len(rows)} cached, {len(missing)} to extract")
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.refresh_judgments:
        judgments.clear()
    with openai_client() as client:
        for index, row in enumerate(missing, start=1):
            print(f"Judge extraction {index}/{len(missing)}: {row['case_id']} | {row['model_profile']} | run_index={row['run_index']}", flush=True)
            judgment = judge_label(client, args.judge_model, row)
            judgments[judgment["row_key"]] = judgment
    write_jsonl(path, sorted(judgments.values(), key=lambda row: (row["case_id"], row["model_profile"], row["run_index"], row["row_key"])))
    print("Judge extraction: done")


def judge_label(client: OpenAI, model: str, row: dict) -> dict:
    if row["error"]:
        return judgment_row(row, "error", "")
    parsed = call_json(
        client,
        model,
        "accent_label_extraction",
        "Extract only the perceived accent label from a model answer. Do not normalize, classify, interpret, or correct the label.",
        extraction_prompt(row["answer"]),
        object_schema(
            {
                "parse_status": {"type": "string", "enum": ["ok", "missing"]},
                "perceived_accent_label": {"type": "string"},
            }
        ),
    )
    label = parsed["perceived_accent_label"].strip()
    status = parsed["parse_status"]
    return judgment_row(row, "missing" if status == "ok" and not label else status, label)


def judgment_row(row: dict, status: str, label: str) -> dict:
    return {
        "row_key": row_key(row),
        "run_id": row["run_id"],
        "run_index": int(row["run_index"]),
        "case_id": row["case_id"],
        "model_profile": row["model_profile"],
        "parse_status": status,
        "perceived_accent_label": label,
    }


def extracted_labels(rows: list[dict], case_ids: set[str], judgments: dict[str, dict]) -> list[str]:
    labels = set()
    for row in rows:
        if row["case_id"] not in case_ids:
            continue
        judgment = judgments[row_key(row)]
        if judgment["parse_status"] == "ok" and judgment["perceived_accent_label"].strip():
            labels.add(judgment["perceived_accent_label"].strip())
    return sorted(labels)


def load_standardization(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8"))["display_labels"])


def ensure_standardization(labels: list[str], standardization: dict[str, str], path: Path, args) -> None:
    target = list(labels) if args.refresh_standardization else [label for label in labels if label not in standardization]
    if not target:
        print(f"Label standardization: all {len(labels)} labels already cached")
        return
    print(f"Label standardization: requesting {len(target)} new label(s); cache has {len(standardization)}")
    existing = [] if args.refresh_standardization else sorted({display for display in standardization.values()})
    with openai_client() as client:
        mapping = judge_standardization(client, args.judge_model, target, existing)
    if args.refresh_standardization:
        standardization.clear()
    standardization.update(mapping)
    write_standardization(path, args.judge_model, standardization)
    print("Label standardization: done")


def judge_standardization(client: OpenAI, model: str, labels: list[str], existing: list[str]) -> dict[str, str]:
    parsed = call_json(
        client,
        model,
        "accent_label_standardization",
        "Standardize extracted accent labels for display. Do not infer intended stimulus conditions or classify correctness.",
        standardization_prompt(labels, existing),
        object_schema(
            {
                "labels": {
                    "type": "array",
                    "items": object_schema({"extracted_label": {"type": "string"}, "display_label": {"type": "string"}}),
                }
            }
        ),
    )
    mapping = {row["extracted_label"]: row["display_label"].strip() for row in parsed["labels"]}
    missing = [label for label in labels if label not in mapping]
    if missing:
        raise ValueError(f"Standardization judge omitted labels: {missing}")
    return mapping


def call_json(client: OpenAI, model: str, name: str, system_prompt: str, user_prompt: str, schema: dict) -> dict:
    def issue() -> dict:
        response = client.responses.create(
            model=model,
            input=[{"role": "developer", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        )
        return json.loads(response.output_text)

    return retry(issue, name)


def retry(func, name: str):
    last_error = None
    for attempt in range(1, JUDGE_ATTEMPTS + 1):
        try:
            return func()
        except Exception as error:
            last_error = error
            if attempt == JUDGE_ATTEMPTS:
                raise
            sleep = JUDGE_SLEEP_SECONDS * (2 ** (attempt - 1)) * (1 + random.random() * 0.3)
            print(f"Judge {name} attempt {attempt}/{JUDGE_ATTEMPTS} failed ({type(error).__name__}: {error}); retrying in {sleep:.1f}s", flush=True)
            time.sleep(sleep)
    raise last_error


def extraction_prompt(answer: str) -> str:
    return (
        "Extract the perceived accent label from this answer.\n"
        "Return the exact label phrase used by the answer, preserving wording as much as possible.\n"
        "If the answer has named fields, use perceived_accent_label.\n"
        "If the answer is a comma-separated list in the requested field order, the first value is the perceived accent label.\n"
        "If no perceived accent label is present, return parse_status='missing' and perceived_accent_label=''.\n\n"
        f"Answer:\n{answer}"
    )


def standardization_prompt(labels: list[str], existing: list[str]) -> str:
    prompt = (
        "You will receive unique perceived accent labels extracted from model answers in one experiment.\n"
        "Return one display label for each extracted label.\n"
        "Group labels only when they are clear wording, capitalization, punctuation, or morphology variants of the same perceived accent label.\n"
        "Preserve specificity. If two labels could plausibly differ in meaning, keep them separate.\n"
        "Do not use the experiment's intended voice accent or text content condition. Do not decide whether a label is correct.\n"
    )
    if existing:
        prompt += f"\nExisting display labels to reuse when appropriate:\n{json.dumps(existing, indent=2)}\n"
    return prompt + f"\nExtracted labels to standardize:\n{json.dumps(labels, indent=2)}"


def object_schema(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}


def load_jsonl_by_key(path: Path, key: str) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row[key]] = row
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def write_standardization(path: Path, model: str, standardization: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"judge_model": model, "display_labels": {label: standardization[label] for label in sorted(standardization)}},
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


def row_key(row: dict) -> str:
    payload = {
        "run_id": row["run_id"],
        "run_index": int(row["run_index"]),
        "case_id": row["case_id"],
        "model_profile": row["model_profile"],
        "answer": row["answer"],
        "error": row["error"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@contextmanager
def openai_client():
    with without_proxy_env():
        yield OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@contextmanager
def without_proxy_env():
    removed = {}
    for name in PROXY_ENV_NAMES:
        if name in os.environ:
            removed[name] = os.environ.pop(name)
    try:
        yield
    finally:
        for name, value in removed.items():
            os.environ[name] = value
