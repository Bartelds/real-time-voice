import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from model_profiles import model_profile

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT
ENV_PATH = REPO_ROOT / ".env"
CASES_PATH = PACKAGE_ROOT / "cases_discrimination_distress.json"
PROMPTS_PATH = PACKAGE_ROOT / "prompts_discrimination.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "runs" / "distress_discrimination" / "_text_floor" / "results.jsonl"


def turn1_transcript() -> str:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    texts = {case["transcript"] for case in cases}
    if len(texts) != 1:
        raise ValueError(f"Turn-1 text is not identical across arms: {sorted(texts)}")
    return texts.pop()


def prompt_fields() -> tuple[str, str]:
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    block = prompts["distress_discrimination"]
    system = "\n".join(block["system"])
    user = "\n".join(block["user"])
    return system, user


def main() -> None:
    parser = argparse.ArgumentParser(prog="distress-discrimination-text-floor")
    parser.add_argument("--model", default="gemini_frontier")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    load_dotenv(ENV_PATH)
    profile = model_profile(args.model)
    transcript = turn1_transcript()
    system, question = prompt_fields()
    user = f'Transcript of what the speaker said: "{transcript}"\n\n{question}'

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    with args.output.open("w", encoding="utf-8") as handle:
        for run_index in range(1, args.runs + 1):
            response = client.models.generate_content(
                model=profile.model,
                config=types.GenerateContentConfig(system_instruction=system),
                contents=user,
            )
            answer = (response.text or "").strip()
            row = {
                "run_index": run_index,
                "task": "distress_discrimination",
                "condition": "words_only",
                "model_profile": profile.name,
                "model": profile.model,
                "modality": "text_only",
                "transcript": transcript,
                "answer": answer,
                "error": None,
            }
            handle.write(json.dumps(row) + "\n")
            print(f"text_floor | {profile.name} | run_index={run_index}: {answer}")
    client.close()
    print(f"Saved words-only floor to {args.output}")


if __name__ == "__main__":
    main()
