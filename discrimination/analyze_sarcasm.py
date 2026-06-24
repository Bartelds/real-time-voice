import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from run import is_refusal

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PACKAGE_ROOT / "cases_discrimination_sarcasm.json"
RUNS_DIR = PACKAGE_ROOT / "runs" / "sarcasm_discrimination"
TEXT_FLOOR = RUNS_DIR / "_text_floor" / "results.jsonl"
MODEL_ORDER = ["gemini_frontier", "openai_realtime", "gemini_live"]


def classify(answer: str, error) -> str:
    if error or is_refusal(answer):
        return "refused"
    s = answer.strip().lower()
    yes = re.search(r"\byes\b", s)
    no = re.search(r"\bno\b", s)
    if yes and (no is None or yes.start() < no.start()):
        return "sarcastic"
    if no:
        return "not"
    if re.search(r"not\s+sarcastic|sounds?\s+sincere|\bsincere\b", s):
        return "not"
    if "sarcastic" in s:
        return "sarcastic"
    return "unparseable"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="sarcasm-discrimination-analyze")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    id_to_group = {case["id"]: case["group"] for case in cases}

    # counts[model][group][label] = count
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for case in cases:
        group = id_to_group[case["id"]]
        for row in read_rows(RUNS_DIR / case["id"] / "results.jsonl"):
            counts[row["model_profile"]][group][classify(row["answer"], row["error"])] += 1

    print("=== words-only floor (Run 1): does the transcript alone signal sarcasm? ===")
    floor = read_rows(TEXT_FLOOR)
    if floor:
        fc: dict[str, int] = defaultdict(int)
        for row in floor:
            fc[classify(row["answer"], row["error"])] += 1
        n = sum(fc.values())
        print(f"  {floor[0]['model_profile']} (text): n={n}  sarcastic={fc['sarcastic']}  not={fc['not']}  "
              f"refused={fc['refused']}  unparseable={fc['unparseable']}  -> sarcastic rate={fc['sarcastic']/n:.2f}")
    else:
        print("  (no text-floor results yet)")

    print("\n=== audio probe: P(sarcastic) by delivery, and separation sarcastic - sincere ===")
    header = f"  {'model':16} {'sincere sarcastic/n':>20} {'sarcastic sarcastic/n':>22} {'refused sincere,sarc':>22} {'separation [95% CI]':>28}"
    print(header)
    models = [m for m in MODEL_ORDER if m in counts] + [m for m in counts if m not in MODEL_ORDER]
    for model in models:
        sincere = counts[model]["sincere"]
        sarcastic = counts[model]["sarcastic"]
        n_sincere = sum(sincere.values())
        n_sarcastic = sum(sarcastic.values())
        p_sincere = sincere["sarcastic"] / n_sincere if n_sincere else float("nan")
        p_sarcastic = sarcastic["sarcastic"] / n_sarcastic if n_sarcastic else float("nan")
        sep = p_sarcastic - p_sincere
        if n_sincere and n_sarcastic:
            se = math.sqrt(p_sincere * (1 - p_sincere) / n_sincere + p_sarcastic * (1 - p_sarcastic) / n_sarcastic)
            half = 1.96 * se
            sep_str = f"{sep:+.2f} [{sep - half:+.2f}, {sep + half:+.2f}]"
        else:
            sep_str = "n/a"
        sincere_cell = f"{sincere['sarcastic']}/{n_sincere}"
        sarcastic_cell = f"{sarcastic['sarcastic']}/{n_sarcastic}"
        ref_cell = f"{sincere['refused']},{sarcastic['refused']}"
        print(f"  {model:16} {sincere_cell:>20} {sarcastic_cell:>22} {ref_cell:>22} {sep_str:>28}")
    print("\nReading: separation CI excluding 0 => the model discriminates the delivery. "
          "High refusal counts make the rate uninterpretable (treated as not-sarcastic here). "
          "n=5/arm pilot uses a Wald CI; switch to a mixed-effects logistic with (1|voice) when scaling across voices.")


if __name__ == "__main__":
    main()
