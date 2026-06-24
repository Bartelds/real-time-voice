import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from run import is_refusal

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PACKAGE_ROOT / "cases_discrimination_distress.json"
RUNS_DIR = PACKAGE_ROOT / "runs" / "distress_discrimination"
TEXT_FLOOR = RUNS_DIR / "_text_floor" / "results.jsonl"
MODEL_ORDER = ["gemini_frontier", "openai_realtime", "gemini_live"]


def classify(answer: str, error) -> str:
    if error or is_refusal(answer):
        return "refused"
    s = answer.strip().lower()
    yes = re.search(r"\byes\b", s)
    no = re.search(r"\bno\b", s)
    if yes and (no is None or yes.start() < no.start()):
        return "distressed"
    if no:
        return "not"
    if re.search(r"not\s+distressed|sounds?\s+calm|\bcalm\b", s):
        return "not"
    if "distressed" in s:
        return "distressed"
    return "unparseable"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="distress-discrimination-analyze")
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

    print("=== words-only floor (Run 1): does the transcript alone signal distress? ===")
    floor = read_rows(TEXT_FLOOR)
    if floor:
        fc: dict[str, int] = defaultdict(int)
        for row in floor:
            fc[classify(row["answer"], row["error"])] += 1
        n = sum(fc.values())
        print(f"  {floor[0]['model_profile']} (text): n={n}  distressed={fc['distressed']}  not={fc['not']}  "
              f"refused={fc['refused']}  unparseable={fc['unparseable']}  -> distressed rate={fc['distressed']/n:.2f}")
    else:
        print("  (no text-floor results yet)")

    print("\n=== audio probe: P(distressed) by delivery, and separation cry - calm ===")
    header = f"  {'model':16} {'calm distressed/n':>18} {'cry distressed/n':>18} {'refused calm,cry':>18} {'separation [95% CI]':>28}"
    print(header)
    models = [m for m in MODEL_ORDER if m in counts] + [m for m in counts if m not in MODEL_ORDER]
    for model in models:
        calm = counts[model]["calm"]
        cry = counts[model]["cry"]
        n_calm = sum(calm.values())
        n_cry = sum(cry.values())
        p_calm = calm["distressed"] / n_calm if n_calm else float("nan")
        p_cry = cry["distressed"] / n_cry if n_cry else float("nan")
        sep = p_cry - p_calm
        if n_calm and n_cry:
            se = math.sqrt(p_calm * (1 - p_calm) / n_calm + p_cry * (1 - p_cry) / n_cry)
            half = 1.96 * se
            sep_str = f"{sep:+.2f} [{sep - half:+.2f}, {sep + half:+.2f}]"
        else:
            sep_str = "n/a"
        calm_cell = f"{calm['distressed']}/{n_calm}"
        cry_cell = f"{cry['distressed']}/{n_cry}"
        ref_cell = f"{calm['refused']},{cry['refused']}"
        print(f"  {model:16} {calm_cell:>18} {cry_cell:>18} {ref_cell:>18} {sep_str:>28}")
    print("\nReading: separation CI excluding 0 => the model discriminates the delivery. "
          "High refusal counts make the rate uninterpretable (treated as not-distressed here). "
          "n=5/arm pilot uses a Wald CI; switch to a mixed-effects logistic with (1|voice) when scaling across voices.")


if __name__ == "__main__":
    main()
