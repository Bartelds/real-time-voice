import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

from openai import OpenAI


DEFAULT_JUDGE_MODEL = "gpt-5.5"
OPENAI_CLIENT_TIMEOUT = 60.0
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
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["row_key"]] = row
    return rows


def ensure_judgments(rows: list[dict], judgments: dict[str, dict], path: Path, args) -> None:
    missing = [row for row in rows if args.refresh_judgments or row_key(row) not in judgments]
    if not missing:
        print(f"Age judge: all {len(rows)} rows already cached")
        return
    print(f"Age judge: {len(rows) - len(missing)}/{len(rows)} cached, {len(missing)} to extract")
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.refresh_judgments:
        judgments.clear()
    with openai_client() as client:
        for index, row in enumerate(missing, start=1):
            print(f"Age judge {index}/{len(missing)}: {row['case_id']} | {row['model_profile']} | run_index={row['run_index']}", flush=True)
            judgment = judge_age(client, args.judge_model, row)
            judgments[judgment["row_key"]] = judgment
    write_jsonl(path, sorted(judgments.values(), key=lambda row: (row["case_id"], row["model_profile"], row["run_index"], row["row_key"])))
    print("Age judge: done")


def judge_age(client: OpenAI, model: str, row: dict) -> dict:
    if row["error"]:
        return judgment_row(row, "error", "", "", None)
    parsed = call_json(
        client,
        model,
        "age_field_extraction",
        "Extract only age-perception fields from a model answer. Do not judge correctness, infer intent, or add interpretation.",
        extraction_prompt(row["answer"]),
        extraction_schema(),
    )
    status = parsed["parse_status"]
    midpoint = parsed["estimated_age_midpoint"]
    if status == "ok" and midpoint is None:
        status = "missing"
    return judgment_row(
        row,
        status,
        parsed["age_group"].strip(),
        parsed["perceived_age_range"].strip(),
        midpoint,
    )


def judgment_row(row: dict, status: str, age_group: str, age_range: str, midpoint: float | None) -> dict:
    return {
        "row_key": row_key(row),
        "run_id": row["run_id"],
        "run_index": int(row["run_index"]),
        "case_id": row["case_id"],
        "model_profile": row["model_profile"],
        "parse_status": status,
        "age_group": age_group,
        "perceived_age_range": age_range,
        "estimated_age_midpoint": midpoint,
    }


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
        "Extract the age perception fields from this answer.\n"
        "Return age_group as the age group label stated by the answer, lightly normalized only for capitalization and spacing.\n"
        "Return perceived_age_range as text, preserving the answer's wording as much as possible.\n"
        "Return estimated_age_midpoint as a number. If the answer explicitly states estimated_age_midpoint, use that value exactly. If no explicit midpoint is stated but the answer gives an age range, use the midpoint of that range. If it gives one estimated age, use that age. If no numeric estimate is present, return null.\n"
        "Return parse_status='ok' if an age perception answer is present, otherwise parse_status='missing'.\n"
        "Do not infer age from transcript content. Use only the model's stated answer.\n\n"
        f"Answer:\n{answer}"
    )


def extraction_schema() -> dict:
    return object_schema(
        {
            "parse_status": {"type": "string", "enum": ["ok", "missing"]},
            "age_group": {"type": "string"},
            "perceived_age_range": {"type": "string"},
            "estimated_age_midpoint": {"type": ["number", "null"]},
        }
    )


def object_schema(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


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
        yield OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=OPENAI_CLIENT_TIMEOUT)


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
