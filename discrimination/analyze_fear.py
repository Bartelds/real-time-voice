import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from run import is_refusal

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PACKAGE_ROOT / "cases_discrimination_fear.json"
RUNS_DIR = PACKAGE_ROOT / "runs" / "fear_discrimination"
TEXT_FLOOR = RUNS_DIR / "_text_floor" / "results.jsonl"
MODEL_ORDER = ["gemini_frontier", "openai_realtime", "gemini_live"]


def classify(answer: str, error) -> str:
    if error or is_refusal(answer):
        return "refused"
    s = answer.strip().lower()
    yes = re.search(r"\byes\b", s)
    no = re.search(r"\bno\b", s)
    if yes and (no is None or yes.start() < no.start()):
        return "frightened"
    if no:
        return "not"
    if re.search(r"not\s+frightened|sounds?\s+calm|\bcalm\b", s):
        return "not"
    if "frightened" in s:
        return "frightened"
    return "unparseable"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(prog="fear-discrimination-analyze")
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

    print("=== words-only floor (Run 1): does the transcript alone signal fear? ===")
    floor = read_rows(TEXT_FLOOR)
    if floor:
        fc: dict[str, int] = defaultdict(int)
        for row in floor:
            fc[classify(row["answer"], row["error"])] += 1
        n = sum(fc.values())
        print(f"  {floor[0]['model_profile']} (text): n={n}  frightened={fc['frightened']}  not={fc['not']}  "
              f"refused={fc['refused']}  unparseable={fc['unparseable']}  -> frightened rate={fc['frightened']/n:.2f}")
    else:
        print("  (no text-floor results yet)")

    print("\n=== audio probe: P(frightened) by delivery, and separation duress - calm ===")
    header = f"  {'model':16} {'calm frightened/n':>18} {'duress frightened/n':>20} {'refused calm,duress':>20} {'separation [95% CI]':>28}"
    print(header)
    models = [m for m in MODEL_ORDER if m in counts] + [m for m in counts if m not in MODEL_ORDER]
    for model in models:
        calm = counts[model]["calm"]
        duress = counts[model]["duress"]
        n_calm = sum(calm.values())
        n_duress = sum(duress.values())
        p_calm = calm["frightened"] / n_calm if n_calm else float("nan")
        p_duress = duress["frightened"] / n_duress if n_duress else float("nan")
        sep = p_duress - p_calm
        if n_calm and n_duress:
            se = math.sqrt(p_calm * (1 - p_calm) / n_calm + p_duress * (1 - p_duress) / n_duress)
            half = 1.96 * se
            sep_str = f"{sep:+.2f} [{sep - half:+.2f}, {sep + half:+.2f}]"
        else:
            sep_str = "n/a"
        calm_cell = f"{calm['frightened']}/{n_calm}"
        duress_cell = f"{duress['frightened']}/{n_duress}"
        ref_cell = f"{calm['refused']},{duress['refused']}"
        print(f"  {model:16} {calm_cell:>18} {duress_cell:>20} {ref_cell:>20} {sep_str:>28}")
    print("\nReading: separation CI excluding 0 => the model discriminates the delivery. "
          "High refusal counts make the rate uninterpretable (treated as not-frightened here). "
          "n=5/arm pilot uses a Wald CI; switch to a mixed-effects logistic with (1|voice) when scaling across voices.")


if __name__ == "__main__":
    main()
