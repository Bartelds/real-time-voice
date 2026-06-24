import argparse
import asyncio
import json
import mimetypes
import os
import random
import re
import subprocess
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from realtime_agents import (
    QwenRealtimeVoiceAgent,
    RealtimeConfig,
    RealtimeVoiceAgent,
    VoiceProfile,
    to_jsonable,
)

from model_profiles import MODEL_PROFILES, ModelProfile, make_run_id, model_profile


REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_CASES_PATH = Path(__file__).with_name("cases.json")
DEFAULT_PROMPTS_PATH = Path(__file__).with_name("prompts.json")
DEFAULT_RUNS_DIR = Path(__file__).with_name("runs")
OPENAI_SAMPLE_RATE = 24000
GEMINI_LIVE_SAMPLE_RATE = 16000
QWEN_REALTIME_SAMPLE_RATE = 16000
QWEN_INPUT_TRANSCRIPTION_MODEL = "gummy-realtime-v1"
SAMPLE_WIDTH_BYTES = 2
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

REFUSAL_PATTERNS = (
    r"(?:actual )?audio (?:file|stream|recording|input|clip)?[^.]{0,40}(?:inaccessible|unavailable|not (?:available|provided|accessible|attached))",
    r"(?:actual )?audio (?:file|stream|recording|input|clip)?[^.]{0,40}(?:was|is|appears|seems) (?:empty|silent|blank|missing|absent|inaudible)",
    r"no (?:actual )?audio[^.]{0,60}(?:provided|available|received|accessible|attached|detected|present|attached|input)",
    r"no (?:actual )?audio (?:input|file|recording|stream|clip)\b",
    r"(?:cannot|can'?t|unable to|do(?:es)? not|don'?t) (?:access|hear|play|process|listen to|detect|perceive|receive)\b[^.]{0,40}(?:audio|recording|clip|speech|sound)",
    r"(?:i|we) do(?:es)? not have access to (?:the )?(?:audio|recording|clip)",
    r"(?:evaluation|evaluat(?:ed|ing)|assessment|assess(?:ed|ing)?|judging|judgment|judg(?:ed|ing)|analysis|analyz(?:ed|ing)|rating|rated) (?:is )?based (?:only |solely |purely )?on (?:the )?(?:provided )?(?:transcript|text|written|content|description|script)",
    r"based (?:only|solely|purely) on (?:the )?(?:provided )?(?:transcript|text|written content|description|script)",
    r"only (?:the )?(?:transcript|text|written (?:content|input)|description|script)[^.]{0,40}(?:was )?(?:provided|available|given)",
    r"(?:transcript|text|written content)[^.]{0,40}(?:only|alone)[^.]{0,60}(?:provided|available|given|received)",
    r"no (?:audible|acoustic)[^.]{0,60}(?:available|provided|present|detected)",
    r"there (?:aren'?t|are no|is no|wasn'?t|was no) (?:reliable )?(?:audible|acoustic|vocal) (?:cues|features|input|signal|information)",
    r"no (?:pronunciation|rhythm|intonation|vowel|consonant)[^.]{0,80}(?:available|provided|detected|present)",
    r"the (?:audio|recording|clip) (?:appears|seems|is)[^.]{0,40}(?:silent|blank|empty|missing|inaudible)",
    r"no (?:speech|voice|sound)[^.]{0,40}(?:detected|present|audible|provided|available) (?:in|on|from) (?:the )?(?:audio|recording|clip)",
)
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), flags=re.IGNORECASE)


@dataclass(frozen=True)
class Case:
    id: str
    task: str
    audio: str
    group: str = ""
    suite: str = "default"
    status: str = "pending"
    pattern: str = ""
    notes: str = ""


def main() -> None:
    load_dotenv(ENV_PATH)
    args = build_parser().parse_args()
    profiles = [model_profile(name) for name in args.models]
    if args.preflight:
        run_preflight(profiles, args)
        return
    run_id = make_run_id()
    cases = selected_cases(load_cases(args.cases), args)
    prompts = load_prompts(args.prompts)
    if not cases:
        raise ValueError("No cases selected")
    validate_audio_paths(cases, args.cases)
    output_root = args.output_root or DEFAULT_RUNS_DIR
    snapshot = {
        "run_id": run_id,
        "cases_path": str(args.cases),
        "prompts_path": str(args.prompts),
        "runs_per_case": args.runs_per_case,
        "target_runs_per_case": args.target_runs_per_case,
        "models": [to_jsonable(profile) for profile in profiles],
        "cases": [to_jsonable(case) for case in cases],
        "prompts": prompts,
    }
    write_run_snapshots(output_root, cases, run_id, snapshot)
    next_run_index, completed_counts = build_run_counters(output_root, cases, profiles)
    jobs = []
    for case in cases:
        for profile in profiles:
            runs_to_schedule = runs_needed_for_case_model(case, profile, completed_counts, args)
            for _ in range(runs_to_schedule):
                next_run_index[(case.id, profile.name)] += 1
                jobs.append((case, profile, next_run_index[(case.id, profile.name)]))
    print(f"Run root: {output_root}")
    print(f"Run id: {run_id}")
    print(f"Cases: {len(cases)} | Models: {len(profiles)} | Jobs: {len(jobs)}")
    for case, profile, run_index in jobs:
        row = run_case_safely(case, profile, run_index, args, run_id, prompts)
        append_result(case_output_dir(output_root, case), row)
        marker = row["error"] if row["answer"] == "" and row["error"] else row["answer"]
        print(f"{case.id} | {profile.name} | run_index={run_index} attempts={row['attempts']}: {marker}")
    print(f"Saved results under {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capability-loss-run")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--status", default=None)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_PROFILES), default=list(MODEL_PROFILES))
    parser.add_argument("--runs-per-case", type=int, default=1)
    parser.add_argument("--target-runs-per-case", type=int, default=None)
    parser.add_argument("--openai-voice", default="cedar")
    parser.add_argument("--qwen-voice", default="Ethan")
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default="high")
    parser.add_argument("--gemini-live-api-version", default="v1alpha")
    parser.add_argument("--timeout-seconds", type=float, default=90)
    parser.add_argument("--max-refusal-attempts", type=int, default=3)
    parser.add_argument("--refusal-retry-sleep-seconds", type=float, default=1.5)
    parser.add_argument("--preflight", action="store_true")
    return parser


def load_cases(path: Path) -> list[Case]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        Case(
            id=row["id"],
            task=row["task"],
            audio=row["audio"],
            group=row["group"] if "group" in row else "",
            suite=row["suite"] if "suite" in row else "default",
            status=row["status"] if "status" in row else "pending",
            pattern=row["pattern"] if "pattern" in row else "",
            notes=row["notes"] if "notes" in row else "",
        )
        for row in rows
    ]
    duplicates = sorted(case_id for case_id, count in Counter(case.id for case in cases).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate case ids in {path}: {duplicates}")
    return cases


def load_prompts(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_cases(cases: list[Case], args: argparse.Namespace) -> list[Case]:
    selected = cases
    if args.case_id is not None:
        selected = [case for case in selected if case.id == args.case_id]
    if args.suite is not None:
        selected = [case for case in selected if case.suite == args.suite]
    if args.task is not None:
        selected = [case for case in selected if case.task == args.task]
    if args.status is not None:
        selected = [case for case in selected if case.status == args.status]
    return selected


def case_output_dir(output_root: Path, case: Case) -> Path:
    if case.task == "accent_perception":
        return output_root / case.task / accent_folder_for_case(case) / case.id
    return output_root / case.task / case.id


def accent_folder_for_case(case: Case) -> str:
    parts = Path(case.audio).parts
    if "accents" not in parts:
        raise ValueError(f"Accent case audio path does not include an accents folder: {case.audio}")
    index = parts.index("accents")
    if len(parts) <= index + 1:
        raise ValueError(f"Accent case audio path does not include an accent subfolder: {case.audio}")
    return parts[index + 1]


def write_run_snapshots(output_root: Path, cases: list[Case], run_id: str, snapshot: dict) -> None:
    for case in cases:
        output_dir = case_output_dir(output_root, case)
        snapshot_dir = output_dir / "run_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / f"{run_id}.json").write_text(
            json.dumps(to_jsonable(snapshot), indent=2) + "\n",
            encoding="utf-8",
        )


def validate_audio_paths(cases: list[Case], cases_path: Path) -> None:
    missing = [
        f"{case.id}: {resolve_audio_path(cases_path, case.audio)}"
        for case in cases
        if not resolve_audio_path(cases_path, case.audio).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing audio files for cases:\n  " + "\n  ".join(missing))


def build_run_counters(
    output_root: Path,
    cases: list[Case],
    profiles: list[ModelProfile],
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    next_indices: dict[tuple[str, str], int] = {(case.id, profile.name): 0 for case in cases for profile in profiles}
    completed_counts: dict[tuple[str, str], int] = {(case.id, profile.name): 0 for case in cases for profile in profiles}
    for case in cases:
        results_path = case_output_dir(output_root, case) / "results.jsonl"
        if not results_path.exists():
            continue
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["case_id"], row["model_profile"])
            if key in next_indices:
                next_indices[key] = max(next_indices[key], int(row["run_index"]))
                if row["error"] is None:
                    completed_counts[key] += 1
    return next_indices, completed_counts


def runs_needed_for_case_model(
    case: Case,
    profile: ModelProfile,
    current_counts: dict[tuple[str, str], int],
    args: argparse.Namespace,
) -> int:
    if args.target_runs_per_case is None:
        return args.runs_per_case
    current = current_counts[(case.id, profile.name)]
    return max(0, args.target_runs_per_case - current)


TRANSIENT_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)


def invoke_provider(case: Case, profile: ModelProfile, audio_path: Path, args: argparse.Namespace, prompts: dict) -> str:
    if profile.provider == "openai_realtime":
        return asyncio.run(
            asyncio.wait_for(
                run_openai_realtime(case, profile, audio_path, args, prompts),
                timeout=args.timeout_seconds,
            )
        )
    if profile.provider == "gemini_live":
        return asyncio.run(
            asyncio.wait_for(
                run_gemini_live(case, profile, audio_path, args, prompts),
                timeout=args.timeout_seconds,
            )
        )
    if profile.provider == "gemini_frontier":
        return run_gemini_frontier(case, profile, audio_path, prompts)
    if profile.provider == "qwen_realtime":
        return asyncio.run(
            asyncio.wait_for(
                run_qwen_realtime(case, profile, audio_path, args, prompts),
                timeout=args.timeout_seconds,
            )
        )
    raise ValueError(f"Unknown provider: {profile.provider}")


def is_refusal(answer: str) -> bool:
    if answer is None:
        return True
    stripped = answer.strip()
    if not stripped:
        return True
    return bool(REFUSAL_RE.search(stripped))


def retry_sleep_seconds(attempt: int, base_sleep: float) -> float:
    return base_sleep * (2 ** (attempt - 1)) + random.uniform(0.0, base_sleep * 0.5)


def run_case(
    case: Case,
    profile: ModelProfile,
    run_index: int,
    args: argparse.Namespace,
    run_id: str,
    prompts: dict,
) -> dict[str, object]:
    audio_path = resolve_audio_path(args.cases, case.audio)
    prior_attempts: list[dict[str, str]] = []
    answer = ""
    attempts = 0
    for attempt in range(1, args.max_refusal_attempts + 1):
        attempts = attempt
        try:
            answer = invoke_provider(case, profile, audio_path, args, prompts)
        except TRANSIENT_RETRY_EXCEPTIONS as error:
            outcome = "transient_error"
            answer = f"[{type(error).__name__}] {error}".strip()
            prior_attempts.append({"outcome": outcome, "answer": answer})
            if attempt < args.max_refusal_attempts:
                time.sleep(retry_sleep_seconds(attempt, args.refusal_retry_sleep_seconds))
                continue
            row = result_row(case, profile, run_index, run_id, answer)
            row["attempts"] = attempts
            row["prior_attempts"] = prior_attempts
            row["error"] = f"transient_error_after_{attempts}_attempts"
            return row
        if not is_refusal(answer):
            row = result_row(case, profile, run_index, run_id, answer)
            row["attempts"] = attempts
            row["prior_attempts"] = prior_attempts
            return row
        prior_attempts.append({"outcome": "refusal", "answer": answer})
        if attempt < args.max_refusal_attempts:
            time.sleep(retry_sleep_seconds(attempt, args.refusal_retry_sleep_seconds))
    row = result_row(case, profile, run_index, run_id, answer)
    row["attempts"] = attempts
    row["prior_attempts"] = prior_attempts[:-1]
    row["error"] = f"refusal_after_{attempts}_attempts"
    return row


def run_case_safely(
    case: Case,
    profile: ModelProfile,
    run_index: int,
    args: argparse.Namespace,
    run_id: str,
    prompts: dict,
) -> dict[str, object]:
    try:
        return run_case(case, profile, run_index, args, run_id, prompts)
    except Exception as error:
        row = result_row(case, profile, run_index, run_id, "")
        row["error"] = "".join(traceback.format_exception_only(type(error), error)).strip()
        return row


def run_preflight(profiles: list[ModelProfile], args: argparse.Namespace) -> None:
    for profile in profiles:
        if profile.provider == "openai_realtime":
            asyncio.run(preflight_openai_realtime(profile, args))
        elif profile.provider == "gemini_live":
            asyncio.run(preflight_gemini_live(profile, args))
        elif profile.provider == "gemini_frontier":
            preflight_gemini_frontier(profile)
        elif profile.provider == "qwen_realtime":
            asyncio.run(preflight_qwen_realtime(profile, args))
        else:
            raise ValueError(f"Unknown provider: {profile.provider}")
        print(f"{profile.name}: ok")


async def preflight_openai_realtime(profile: ModelProfile, args: argparse.Namespace) -> None:
    config = RealtimeConfig(
        model=profile.model,
        output_modalities=["audio"],
        reasoning_effort=args.reasoning_effort,
    )
    agent = RealtimeVoiceAgent(
        player_id="preflight",
        recipient_id="speaker",
        instructions="Say ok.",
        voice_profile=VoiceProfile(voice=args.openai_voice),
        realtime_config=config,
    )
    await asyncio.wait_for(agent.connect(api_key=required_env("OPENAI_API_KEY")), timeout=args.timeout_seconds)
    await agent.close()


def preflight_gemini_frontier(profile: ModelProfile) -> None:
    client = genai.Client(api_key=required_env("GOOGLE_API_KEY"))
    try:
        response = client.models.generate_content(
            model=profile.model,
            contents="Say ok.",
        )
    finally:
        client.close()
    if response.text is None:
        raise RuntimeError(f"{profile.name} returned no text in preflight")


async def preflight_gemini_live(profile: ModelProfile, args: argparse.Namespace) -> None:
    with without_proxy_env():
        client = genai.Client(
            api_key=required_env("GOOGLE_API_KEY"),
            http_options=types.HttpOptions(api_version=args.gemini_live_api_version),
        )
        try:
            config = types.LiveConnectConfig(
                response_modalities=[types.Modality.AUDIO],
                output_audio_transcription=types.AudioTranscriptionConfig(),
            )
            async with client.aio.live.connect(model=profile.model, config=config) as session:
                await session.send_realtime_input(text="Say ok.")
                text = await collect_gemini_live_transcript(session, args.timeout_seconds)
        finally:
            client.close()
    if not text:
        raise RuntimeError(f"{profile.name} returned no transcript in preflight")


async def run_openai_realtime(
    case: Case,
    profile: ModelProfile,
    audio_path: Path,
    args: argparse.Namespace,
    prompts: dict,
) -> str:
    config = RealtimeConfig(
        model=profile.model,
        output_modalities=["audio"],
        reasoning_effort=args.reasoning_effort,
    )
    agent = RealtimeVoiceAgent(
        player_id="audio_judge",
        recipient_id="speaker",
        instructions=prompt_for(prompts, case.task, "system"),
        voice_profile=VoiceProfile(voice=args.openai_voice),
        realtime_config=config,
    )
    await agent.connect(api_key=required_env("OPENAI_API_KEY"))
    try:
        audio = await asyncio.to_thread(read_audio_pcm, audio_path, OPENAI_SAMPLE_RATE)
        turn = await agent.reply_to_audio(audio, text=prompt_for(prompts, case.task, "user"))
        return turn.transcript
    finally:
        await agent.close()


async def preflight_qwen_realtime(profile: ModelProfile, args: argparse.Namespace) -> None:
    agent = build_qwen_agent(profile, "Say ok.", args)
    await asyncio.wait_for(agent.connect(api_key=required_env("DASHSCOPE_API_KEY")), timeout=args.timeout_seconds)
    await agent.close()


def build_qwen_agent(profile: ModelProfile, instructions: str, args: argparse.Namespace) -> QwenRealtimeVoiceAgent:
    if profile.websocket_url is None:
        raise ValueError(f"Profile {profile.name} is missing websocket_url")
    return QwenRealtimeVoiceAgent(
        player_id="audio_judge",
        recipient_id="speaker",
        instructions=instructions,
        voice=args.qwen_voice,
        model=profile.model,
        websocket_url=profile.websocket_url,
        input_audio_transcription_model=QWEN_INPUT_TRANSCRIPTION_MODEL,
    )


async def run_qwen_realtime(
    case: Case,
    profile: ModelProfile,
    audio_path: Path,
    args: argparse.Namespace,
    prompts: dict,
) -> str:
    agent = build_qwen_agent(profile, combined_system_instruction(prompts, case.task), args)
    await agent.connect(api_key=required_env("DASHSCOPE_API_KEY"))
    try:
        audio = await asyncio.to_thread(read_audio_pcm, audio_path, QWEN_REALTIME_SAMPLE_RATE)
        turn = await agent.reply_to_audio(audio)
        return turn.transcript
    finally:
        await agent.close()


def run_gemini_frontier(case: Case, profile: ModelProfile, audio_path: Path, prompts: dict) -> str:
    client = genai.Client(api_key=required_env("GOOGLE_API_KEY"))
    try:
        response = client.models.generate_content(
            model=profile.model,
            config=types.GenerateContentConfig(
                system_instruction=prompt_for(prompts, case.task, "system"),
            ),
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=audio_path.read_bytes(),
                            mime_type=mime_type_for(audio_path),
                        ),
                        types.Part.from_text(text=prompt_for(prompts, case.task, "user")),
                    ],
                )
            ],
        )
        return response.text or ""
    finally:
        client.close()


async def run_gemini_live(
    case: Case,
    profile: ModelProfile,
    audio_path: Path,
    args: argparse.Namespace,
    prompts: dict,
) -> str:
    with without_proxy_env():
        client = genai.Client(
            api_key=required_env("GOOGLE_API_KEY"),
            http_options=types.HttpOptions(api_version=args.gemini_live_api_version),
        )
        try:
            pcm = await asyncio.to_thread(read_audio_pcm, audio_path, GEMINI_LIVE_SAMPLE_RATE)
            config = types.LiveConnectConfig(
                response_modalities=[types.Modality.AUDIO],
                output_audio_transcription=types.AudioTranscriptionConfig(),
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
                system_instruction=gemini_text_content(combined_system_instruction(prompts, case.task)),
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
                    turn_coverage=types.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
                ),
            )
            async with client.aio.live.connect(model=profile.model, config=config) as session:
                await session.send_realtime_input(activity_start=types.ActivityStart())
                await session.send_realtime_input(
                    audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={GEMINI_LIVE_SAMPLE_RATE}")
                )
                await session.send_realtime_input(activity_end=types.ActivityEnd())
                transcript = await collect_gemini_live_transcript(session, args.timeout_seconds)
            return transcript.strip()
        finally:
            client.close()


def result_row(
    case: Case,
    profile: ModelProfile,
    run_index: int,
    run_id: str,
    answer: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "run_index": run_index,
        "case_id": case.id,
        "task": case.task,
        "model_profile": profile.name,
        "model": profile.model,
        "answer": answer,
        "attempts": 1,
        "prior_attempts": [],
        "error": None,
    }


def prompt_for(prompts: dict, task: str, field: str) -> str:
    if task not in prompts:
        raise ValueError(f"No prompts are defined for task: {task}")
    if field not in prompts[task]:
        raise ValueError(f"No {field} prompt is defined for task: {task}")
    value = prompts[task][field]
    if isinstance(value, list):
        return "\n".join(value)
    return value


def combined_system_instruction(prompts: dict, task: str) -> str:
    return prompt_for(prompts, task, "system") + "\n\n" + prompt_for(prompts, task, "user")


def gemini_text_content(text: str) -> types.Content:
    return types.Content(parts=[types.Part.from_text(text=text)])


async def collect_gemini_live_transcript(session, timeout_seconds: float) -> str:
    async def collect() -> str:
        text_parts = []
        async for message in session.receive():
            if message.server_content:
                transcription = message.server_content.output_transcription
                if transcription and transcription.text:
                    text_parts.append(transcription.text)
                if message.server_content.turn_complete:
                    return "".join(text_parts)
        return "".join(text_parts)

    return await asyncio.wait_for(collect(), timeout=timeout_seconds)


def required_env(name: str) -> str:
    if name not in os.environ:
        raise RuntimeError(f"Missing {name}. Add it to {ENV_PATH} or export it before running.")
    return os.environ[name]


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


def resolve_audio_path(cases_path: Path, audio: str) -> Path:
    path = Path(audio)
    if path.is_absolute():
        return path
    return cases_path.parent / path


def read_audio_pcm(path: Path, sample_rate: int) -> bytes:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        "-f",
        "s16le",
        "-",
    ]
    pcm = subprocess.check_output(command)
    if not pcm:
        raise ValueError(f"Converted audio is empty: {path}")
    if len(pcm) % SAMPLE_WIDTH_BYTES != 0:
        raise ValueError(f"Converted audio byte length is not 16-bit aligned: {path}")
    return pcm


def mime_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed is not None:
        return guessed
    if path.suffix.lower() == ".wav":
        return "audio/wav"
    if path.suffix.lower() == ".mp3":
        return "audio/mpeg"
    if path.suffix.lower() == ".m4a":
        return "audio/mp4"
    return "application/octet-stream"


def append_result(output_dir: Path, row: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(to_jsonable(row)) + "\n")


if __name__ == "__main__":
    main()
