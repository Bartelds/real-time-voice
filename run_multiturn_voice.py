import argparse
import asyncio
import json
import os
import re
import subprocess
import traceback
import urllib.error
import urllib.request
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

from realtime_agents import (
    ProviderTurn,
    QwenRealtimeVoiceAgent,
    RealtimeConfig,
    RealtimeVoiceAgent,
    VoiceProfile,
)

from model_profiles import REALTIME_MODEL_PROFILE_NAMES, ModelProfile, make_run_id, model_profile


REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_STIMULI_DIR = Path(__file__).with_name("stimuli") / "multiturn"
DEFAULT_RUNS_DIR = Path(__file__).with_name("runs")
DEFAULT_EXPERIMENT_ID = "welfare_callback"
TASK = "multiturn"
OPENAI_SAMPLE_RATE = 24000
GEMINI_SAMPLE_RATE = 16000
QWEN_SAMPLE_RATE = 16000
QWEN_INPUT_TRANSCRIPTION_MODEL = "gummy-realtime-v1"
OUTPUT_SAMPLE_RATE = 24000
SAMPLE_WIDTH_BYTES = 2
CONTROLLER_MODEL = "gpt-5.5"
PERCEPTION_PROBE_MAX_ATTEMPTS = 3
PROBE_VALID_RE = re.compile(r"intensity\s*[=:]\s*\d", re.IGNORECASE)
PROBE_REFUSAL_RE = re.compile(
    r"not assessable|text[-\s]?only|no audio|\btranscript\b|without (?:the )?audio|no (?:recording|sound)|can(?:no|')t (?:assess|evaluate|determine|tell)",
    re.IGNORECASE,
)
OPENAI_CLIENT_TIMEOUT = 60.0
ELEVENLABS_MODEL_ID = "eleven_v3"
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"
ELEVENLABS_STABILITY = 0.0
ELEVENLABS_SIMILARITY_BOOST = 0.35
ELEVENLABS_STYLE = 0.0
ELEVENLABS_USE_SPEAKER_BOOST = False
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


@dataclass(frozen=True)
class MultiturnCase:
    case_id: str
    scenario_name: str
    condition: str
    hidden_state: str
    persona: str
    caller_voice_name: str
    caller_voice_id: str | None
    caller_audio: list[str]
    caller_texts: list[str]


@dataclass(frozen=True)
class MultiturnExperiment:
    experiment_id: str
    stimuli_dir: Path
    max_caller_turns: int
    history_format: str
    opening_line: str
    system_instruction: str
    opening_prompt_template: str
    controller_system_prompt: str
    controller_user_prompt_template: str
    perception_probe: str | None
    caller_script: bool
    cases: list[MultiturnCase]


@dataclass(frozen=True)
class ControllerDecision:
    turn_number: int
    decision: str
    reason: str
    next_caller_text: str
    prompt: dict
    stopped_by: str
    forced: bool = False


def main() -> None:
    load_dotenv(ENV_PATH)
    args = build_parser().parse_args()
    if args.list_experiments:
        list_experiments(args.stimuli_dir)
        return
    experiment = load_experiment(args.stimuli_dir, args.experiment_id)
    cases = experiment.cases
    if args.case_id is not None:
        cases = [case for case in cases if case.case_id == args.case_id]
    if args.condition is not None:
        cases = [case for case in cases if case.condition == args.condition]
    if not cases:
        raise ValueError("No multiturn cases selected")
    if args.list_cases:
        for case in cases:
            print(f"{case.case_id}\t{case.scenario_name}\t{case.caller_voice_name}\t{', '.join(case.caller_audio)}")
        return
    profiles = [model_profile(name) for name in args.models]
    run_id = make_run_id()
    output_root = args.output_root or DEFAULT_RUNS_DIR
    output_dir = experiment_output_dir(output_root, experiment)
    next_indices, completed_counts = build_run_counters(output_dir, cases, profiles)
    jobs = []
    for case in cases:
        for profile in profiles:
            runs_to_schedule = runs_needed(case, profile, completed_counts, args)
            for _ in range(runs_to_schedule):
                next_indices[(case.case_id, profile.name)] += 1
                jobs.append((case, profile, next_indices[(case.case_id, profile.name)]))
    cases_to_run = unique_cases_for_jobs(jobs)
    validate_cases(cases_to_run, experiment.stimuli_dir)
    validate_synthesis_voice_ids(cases_to_run, args)
    print(f"Run root: {output_dir}")
    print(f"Experiment: {experiment.experiment_id}")
    print(f"Run id: {run_id}")
    print(f"Cases: {len(cases)} | Models: {len(profiles)} | Jobs: {len(jobs)}")
    for case, profile, run_index in jobs:
        row = run_case_safely(experiment, case, profile, run_index, run_id, output_dir, args)
        append_result(output_dir, row)
        marker = row["error"] if row["error"] else row["agent_response_transcript"]
        probe_suffix = f" | probe: {row['perception_probe_transcript']}" if row["perception_probe_transcript"] else ""
        print(f"{case.case_id} | {profile.name} | run_index={run_index}: {marker}{probe_suffix}")
    print(f"Saved results under {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capability-loss-multiturn")
    parser.add_argument("--stimuli-dir", type=Path, default=DEFAULT_STIMULI_DIR)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--models", nargs="+", choices=REALTIME_MODEL_PROFILE_NAMES, default=list(REALTIME_MODEL_PROFILE_NAMES))
    parser.add_argument("--runs-per-case", type=int, default=1)
    parser.add_argument("--target-runs-per-case", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--openai-voice", default="cedar")
    parser.add_argument("--qwen-voice", default="Ethan")
    parser.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default="high")
    parser.add_argument("--gemini-live-api-version", default="v1alpha")
    parser.add_argument("--controller-model", default=CONTROLLER_MODEL)
    parser.add_argument("--elevenlabs-voice-id", default=None)
    parser.add_argument("--elevenlabs-model-id", default=ELEVENLABS_MODEL_ID)
    parser.add_argument("--elevenlabs-output-format", default=ELEVENLABS_OUTPUT_FORMAT)
    parser.add_argument("--elevenlabs-stability", type=float, default=ELEVENLABS_STABILITY)
    parser.add_argument("--elevenlabs-similarity-boost", type=float, default=ELEVENLABS_SIMILARITY_BOOST)
    parser.add_argument("--elevenlabs-style", type=float, default=ELEVENLABS_STYLE)
    parser.add_argument("--elevenlabs-use-speaker-boost", action=argparse.BooleanOptionalAction, default=ELEVENLABS_USE_SPEAKER_BOOST)
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    return parser


def list_experiments(stimuli_root: Path) -> None:
    for path in sorted(stimuli_root.iterdir()):
        if path.is_dir() and (path / "experiment.json").exists():
            print(path.name)


def load_experiment(stimuli_root: Path, experiment_id: str) -> MultiturnExperiment:
    validate_identifier(experiment_id, "experiment_id")
    stimuli_dir = stimuli_root / experiment_id
    path = stimuli_dir / "experiment.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["experiment_id"] != experiment_id:
        raise ValueError(f"Experiment manifest id {data['experiment_id']} does not match requested id {experiment_id}")
    validate_experiment_manifest(data, path)
    cases = [
        MultiturnCase(
            case_id=row["id"],
            scenario_name=row["name"],
            condition=row["id"],
            hidden_state=row["hidden_state"],
            persona=row["persona"],
            caller_voice_name=row["voice_name"],
            caller_voice_id=row["voice_id"],
            caller_audio=scenario_caller_audio(row),
            caller_texts=scenario_caller_texts(row),
        )
        for row in data["scenarios"]
    ]
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"Duplicate case_id values in {path}")
    for case in cases:
        validate_identifier(case.case_id, "case_id")
        validate_identifier(case.condition, "condition")
    conversation = data["conversation"]
    controller = data["controller"]
    agent = data["agent"]
    perception_probe = agent["perception_probe"] if "perception_probe" in agent else None
    caller_script = data["caller_script"] if "caller_script" in data else False
    return MultiturnExperiment(
        experiment_id=data["experiment_id"],
        stimuli_dir=stimuli_dir,
        max_caller_turns=conversation["max_caller_turns"],
        history_format=conversation["history_format"],
        opening_line=agent["opening_line"],
        system_instruction=agent["system_instruction"],
        opening_prompt_template=agent["opening_prompt_template"],
        controller_system_prompt=controller["system_prompt"],
        controller_user_prompt_template=controller["user_prompt_template"],
        perception_probe=perception_probe,
        caller_script=caller_script,
        cases=cases,
    )


def scenario_caller_audio(row: dict) -> list[str]:
    if "caller_turns" in row:
        return [turn["audio"] for turn in row["caller_turns"]]
    return [row["fixed_audio"]]


def scenario_caller_texts(row: dict) -> list[str]:
    if "caller_turns" in row:
        return [turn["text"] for turn in row["caller_turns"]]
    return []


def validate_experiment_manifest(data: dict, path: Path) -> None:
    validate_identifier(data["experiment_id"], "experiment_id")
    validate_identifier(data["domain"], "domain")
    validate_identifier(data["interaction_type"], "interaction_type")
    data["description"]
    conversation = data["conversation"]
    if not isinstance(conversation["max_caller_turns"], int) or conversation["max_caller_turns"] < 1:
        raise ValueError(f"conversation.max_caller_turns must be a positive integer in {path}")
    conversation["history_format"]
    data["agent"]["opening_line"]
    data["agent"]["system_instruction"]
    data["agent"]["opening_prompt_template"]
    controller = data["controller"]
    controller["system_prompt"]
    controller["user_prompt_template"]
    for row in data["scenarios"]:
        validate_identifier(row["id"], "scenario id")
        row["persona"]
        row["hidden_state"]
        row["name"]
        voice_name = row["voice_name"]
        voice_token = voice_name.lower()
        for field_name, field_value in (
            ("experiment_id", data["experiment_id"]),
            ("scenario id", row["id"]),
        ):
            if voice_token in field_value.lower():
                raise ValueError(f"{field_name} must not contain voice name {voice_name}: {field_value}")
        if "caller_turns" in row:
            if not row["caller_turns"]:
                raise ValueError(f"caller_turns must be a non-empty list in {path}: {row['id']}")
            for turn in row["caller_turns"]:
                turn["text"]
                validate_audio_manifest_path(turn["audio"], voice_name, path)
        else:
            validate_audio_manifest_path(row["fixed_audio"], voice_name, path)
        row["voice_id"]


def validate_audio_manifest_path(audio_path: str, voice_name: str, manifest_path: Path) -> None:
    path = Path(audio_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Audio path must be relative and stay inside the experiment folder in {manifest_path}: {audio_path}")
    if path.name != audio_path:
        raise ValueError(f"Active caller audio must live directly in the experiment folder in {manifest_path}: {audio_path}")
    expected_suffix = "_" + voice_name.lower().replace(" ", "_")
    if not path.stem.lower().endswith(expected_suffix):
        raise ValueError(f"Audio filename must end with voice name suffix {expected_suffix} in {manifest_path}: {audio_path}")


def validate_identifier(value: str, field_name: str) -> None:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if not value or any(character not in allowed for character in value):
        raise ValueError(f"{field_name} must contain only letters, numbers, underscores, and hyphens: {value}")


def validate_cases(cases: list[MultiturnCase], stimuli_dir: Path) -> None:
    missing = [
        str(stimuli_dir / caller_audio)
        for case in cases
        for caller_audio in case.caller_audio
        if not (stimuli_dir / caller_audio).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing caller audio files:\n  " + "\n  ".join(missing))


def unique_cases_for_jobs(jobs: list[tuple[MultiturnCase, ModelProfile, int]]) -> list[MultiturnCase]:
    by_case_id = {}
    for case, _, _ in jobs:
        by_case_id[case.case_id] = case
    return list(by_case_id.values())


def validate_synthesis_voice_ids(cases: list[MultiturnCase], args: argparse.Namespace) -> None:
    if args.elevenlabs_voice_id is not None:
        return
    missing = [case.case_id for case in cases if case.caller_voice_id is None]
    if missing:
        raise ValueError(
            "Missing ElevenLabs voice id for dynamic caller turn synthesis. "
            "Add voice_id in experiment.json or pass --elevenlabs-voice-id for cases: " + ", ".join(missing)
        )


def build_run_counters(
    output_dir: Path,
    cases: list[MultiturnCase],
    profiles: list[ModelProfile],
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    next_indices = {(case.case_id, profile.name): 0 for case in cases for profile in profiles}
    completed_counts = {(case.case_id, profile.name): 0 for case in cases for profile in profiles}
    results_path = output_dir / "results.jsonl"
    if not results_path.exists():
        return next_indices, completed_counts
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


def runs_needed(
    case: MultiturnCase,
    profile: ModelProfile,
    current_counts: dict[tuple[str, str], int],
    args: argparse.Namespace,
) -> int:
    if args.target_runs_per_case is None:
        return args.runs_per_case
    current = current_counts[(case.case_id, profile.name)]
    return max(0, args.target_runs_per_case - current)


def log_stage(case: MultiturnCase, profile: ModelProfile, run_index: int, stage: str) -> None:
    print(f"[{case.case_id} | {profile.name} | run_index={run_index}] {stage}", flush=True)


def run_case_safely(
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    profile: ModelProfile,
    run_index: int,
    run_id: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    try:
        log_stage(case, profile, run_index, "start")
        return asyncio.run(
            asyncio.wait_for(
                run_case(experiment, case, profile, run_index, run_id, output_dir, args),
                timeout=args.timeout_seconds,
            )
        )
    except Exception as error:
        return {
            "run_id": run_id,
            "run_index": run_index,
            "experiment_id": experiment.experiment_id,
            "case_id": case.case_id,
            "scenario_name": case.scenario_name,
            "task": TASK,
            "condition": case.condition,
            "hidden_state": case.hidden_state,
            "persona": case.persona,
            "caller_voice_name": case.caller_voice_name,
            "caller_voice_id": case.caller_voice_id,
            "model_profile": profile.name,
            "model": profile.model,
            "max_caller_turns": experiment.max_caller_turns,
            "agent_opening_prompt": agent_opening_prompt(experiment),
            "agent_opening_transcript": "",
            "artifact_dir": "",
            "agent_opening_audio_path": "",
            "caller_audio": case.caller_audio,
            "agent_response_transcript": "",
            "agent_response_audio_path": "",
            "response_turns": [],
            "generated_caller_text": "",
            "generated_caller_audio_path": "",
            "generated_caller_tts_request_path": "",
            "controller_decisions": [],
            "stopped_by": "error",
            "agent_opening_raw_events_path": "",
            "agent_response_raw_events_path": "",
            "conversation_path": "",
            "conversation_audio_path": "",
            "conversation_transcript_path": "",
            "perception_probe_prompt": "",
            "perception_probe_transcript": "",
            "perception_probe_refused": False,
            "perception_probe_audio_path": "",
            "perception_probe_raw_events_path": "",
            "error": "".join(traceback.format_exception_only(type(error), error)).strip(),
        }


async def run_case(
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    profile: ModelProfile,
    run_index: int,
    run_id: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    if profile.provider == "openai_realtime":
        return await run_openai(experiment, case, profile, run_index, run_id, output_dir, args)
    if profile.provider == "gemini_live":
        return await run_gemini(experiment, case, profile, run_index, run_id, output_dir, args)
    if profile.provider == "qwen_realtime":
        return await run_qwen(experiment, case, profile, run_index, run_id, output_dir, args)
    raise ValueError(f"Unsupported realtime provider: {profile.provider}")


async def run_openai(
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    profile: ModelProfile,
    run_index: int,
    run_id: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    config = RealtimeConfig(model=profile.model, output_modalities=["audio"], reasoning_effort=args.reasoning_effort)
    agent = RealtimeVoiceAgent(
        player_id="agent",
        recipient_id="caller",
        instructions=experiment.system_instruction,
        voice_profile=VoiceProfile(voice=args.openai_voice),
        realtime_config=config,
    )
    await agent.connect(api_key=required_env("OPENAI_API_KEY"))
    try:
        log_stage(case, profile, run_index, "model opening")
        opening = await agent.reply_to_text(agent_opening_prompt(experiment))
        artifact_dir = run_artifact_dir(output_dir, run_id, profile.name, case, run_index)

        async def send_caller_audio(audio_path: Path) -> ProviderTurn:
            caller_pcm = await asyncio.to_thread(read_audio_pcm, audio_path, OPENAI_SAMPLE_RATE)
            return await agent.reply_to_audio(caller_pcm)

        if experiment.caller_script:
            caller_audio_turns, responses, controller_decisions = await run_scripted_conversation_loop(
                experiment,
                case,
                profile,
                run_index,
                output_dir,
                opening,
                send_caller_audio,
            )
        else:
            caller_audio_turns, responses, controller_decisions = await run_conversation_loop(
                experiment,
                case,
                profile,
                run_index,
                output_dir,
                artifact_dir,
                args,
                opening,
                send_caller_audio,
            )
        probe = None
        if experiment.perception_probe:
            log_stage(case, profile, run_index, "perception probe")
            probe = await run_perception_probe(agent.reply_to_text, experiment.perception_probe, PERCEPTION_PROBE_MAX_ATTEMPTS)
    finally:
        await agent.close()
    log_stage(case, profile, run_index, "saving artifacts")
    artifacts = write_artifacts(output_dir, artifact_dir, experiment, case, profile, run_index, run_id, opening, caller_audio_turns, responses, controller_decisions, probe)
    return result_row(experiment, case, profile, run_index, run_id, opening, responses, artifacts, probe)


async def run_conversation_loop(
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    profile: ModelProfile,
    run_index: int,
    output_dir: Path,
    artifact_dir: Path,
    args: argparse.Namespace,
    opening: ProviderTurn,
    send_caller_audio,
) -> tuple[list[dict], list[ProviderTurn], list[ControllerDecision]]:
    caller_audio_turns = [
        {
            "turn_number": 1,
            "caller_audio": case.caller_audio[0],
            "caller_audio_source": "stimulus",
            "generated_caller_text": "",
            "generated_caller_text_path": "",
            "generated_caller_tts_request_path": "",
        }
    ]
    responses = []
    controller_decisions = []
    while True:
        caller_turn = caller_audio_turns[-1]
        turn_number = caller_turn["turn_number"]
        caller_audio_path = resolve_caller_audio_path(output_dir, experiment.stimuli_dir, caller_turn)
        log_stage(case, profile, run_index, f"caller turn {turn_number} audio: {caller_turn['caller_audio']}")
        responses.append(await send_caller_audio(caller_audio_path))
        if not responses[-1].transcript.strip():
            controller_decisions.append(agent_empty_decision(turn_number))
            break
        if turn_number >= experiment.max_caller_turns:
            controller_decisions.append(forced_stop_decision(turn_number, experiment.max_caller_turns))
            break
        log_stage(case, profile, run_index, f"controller after agent response {turn_number}")
        decision = await asyncio.to_thread(
            generate_controller_decision,
            experiment,
            case,
            turn_number,
            opening,
            caller_audio_turns,
            responses,
            args.controller_model,
        )
        controller_decisions.append(decision)
        if decision.decision == "stop":
            break
        next_turn_number = turn_number + 1
        log_stage(case, profile, run_index, f"ElevenLabs caller turn {next_turn_number} synthesis")
        generated_audio = await asyncio.to_thread(
            synthesize_caller_turn,
            output_dir,
            artifact_dir,
            case,
            decision.next_caller_text,
            next_turn_number,
            args,
        )
        caller_audio_turns.append(
            {
                "turn_number": next_turn_number,
                "caller_audio": generated_audio["audio_path"],
                "caller_audio_source": "artifact",
                "generated_caller_text": decision.next_caller_text,
                "generated_caller_text_path": generated_audio["text_path"],
                "generated_caller_tts_request_path": generated_audio["tts_request_path"],
            }
        )
    return caller_audio_turns, responses, controller_decisions


def scripted_decision(turn_number: int, is_last: bool) -> ControllerDecision:
    return ControllerDecision(
        turn_number=turn_number,
        decision="stop" if is_last else "continue",
        reason="scripted_final_turn" if is_last else "scripted_turn",
        next_caller_text="",
        prompt={},
        stopped_by="scripted_end" if is_last else "scripted",
        forced=True,
    )


async def run_scripted_conversation_loop(
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    profile: ModelProfile,
    run_index: int,
    output_dir: Path,
    opening: ProviderTurn,
    send_caller_audio,
) -> tuple[list[dict], list[ProviderTurn], list[ControllerDecision]]:
    caller_audio_turns = []
    responses = []
    controller_decisions = []
    num_turns = len(case.caller_audio)
    for index in range(num_turns):
        turn_number = index + 1
        caller_turn = {
            "turn_number": turn_number,
            "caller_audio": case.caller_audio[index],
            "caller_audio_source": "stimulus",
            "generated_caller_text": case.caller_texts[index] if case.caller_texts else "",
            "generated_caller_text_path": "",
            "generated_caller_tts_request_path": "",
        }
        caller_audio_turns.append(caller_turn)
        caller_audio_path = resolve_caller_audio_path(output_dir, experiment.stimuli_dir, caller_turn)
        log_stage(case, profile, run_index, f"scripted caller turn {turn_number}: {caller_turn['caller_audio']}")
        responses.append(await send_caller_audio(caller_audio_path))
        is_last = turn_number >= num_turns
        if not responses[-1].transcript.strip():
            controller_decisions.append(agent_empty_decision(turn_number))
            break
        controller_decisions.append(scripted_decision(turn_number, is_last))
        if is_last:
            break
    return caller_audio_turns, responses, controller_decisions


def write_artifacts(
    output_dir: Path,
    artifact_dir: Path,
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    profile: ModelProfile,
    run_index: int,
    run_id: str,
    opening: ProviderTurn,
    caller_audio_turns: list[dict],
    responses: list[ProviderTurn],
    controller_decisions: list[ControllerDecision],
    probe: ProviderTurn | None = None,
) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    opening_audio_path = write_turn_audio(output_dir, artifact_dir, "agent_opening", opening)
    opening_raw_events_path = write_raw_events(output_dir, artifact_dir, "agent_opening", opening.raw_events)
    response_turns = []
    if len(responses) != len(caller_audio_turns):
        raise ValueError(f"Expected {len(caller_audio_turns)} agent responses, got {len(responses)}")
    if len(controller_decisions) != len(responses):
        raise ValueError(f"Expected {len(responses)} controller decisions, got {len(controller_decisions)}")
    for caller_audio, response, controller_decision in zip(caller_audio_turns, responses, controller_decisions):
        turn_number = caller_audio["turn_number"]
        response_audio_path = write_turn_audio(output_dir, artifact_dir, f"agent_response_{turn_number}", response)
        response_raw_events_path = write_raw_events(output_dir, artifact_dir, f"agent_response_{turn_number}", response.raw_events)
        controller_prompt_path = ""
        if controller_decision.prompt:
            controller_prompt_path = write_controller_prompt(output_dir, artifact_dir, turn_number, controller_decision.prompt)
        response_turns.append(
            {
                "turn_number": turn_number,
                "caller_audio": caller_audio["caller_audio"],
                "caller_audio_source": caller_audio["caller_audio_source"],
                "generated_caller_text": caller_audio["generated_caller_text"],
                "generated_caller_text_path": caller_audio["generated_caller_text_path"],
                "generated_caller_tts_request_path": caller_audio["generated_caller_tts_request_path"],
                "agent_response_transcript": response.transcript,
                "agent_response_audio_path": response_audio_path,
                "agent_response_raw_events_path": response_raw_events_path,
                "controller_decision": controller_decision_to_jsonable(controller_decision),
                "controller_prompt_path": controller_prompt_path,
            }
        )
    perception_probe_audio_path = ""
    perception_probe_raw_events_path = ""
    if probe is not None:
        perception_probe_audio_path = write_turn_audio(output_dir, artifact_dir, "agent_perception_probe", probe)
        perception_probe_raw_events_path = write_raw_events(output_dir, artifact_dir, "agent_perception_probe", probe.raw_events)
    conversation_audio_path = write_conversation_audio(output_dir, artifact_dir, experiment.stimuli_dir, opening, response_turns, responses)
    conversation_transcript_path = write_conversation_transcript(
        output_dir,
        artifact_dir,
        experiment,
        case,
        opening,
        response_turns,
        responses,
        probe,
    )
    conversation_path = write_conversation(
        output_dir,
        artifact_dir,
        experiment,
        case,
        profile,
        run_index,
        run_id,
        opening,
        responses,
        opening_audio_path,
        opening_raw_events_path,
        response_turns,
        conversation_audio_path,
        conversation_transcript_path,
        probe,
        perception_probe_audio_path,
        perception_probe_raw_events_path,
    )
    latest_generated = latest_generated_caller_turn(response_turns)
    return {
        "artifact_dir": str(artifact_dir.relative_to(output_dir)),
        "agent_opening_audio_path": opening_audio_path,
        "agent_opening_raw_events_path": opening_raw_events_path,
        "response_turns": response_turns,
        "generated_caller_text": latest_generated["generated_caller_text"] if latest_generated else "",
        "generated_caller_audio_path": latest_generated["caller_audio"] if latest_generated else "",
        "generated_caller_tts_request_path": latest_generated["generated_caller_tts_request_path"] if latest_generated else "",
        "controller_decisions": [controller_decision_to_jsonable(decision) for decision in controller_decisions],
        "stopped_by": stopped_by(controller_decisions),
        "conversation_path": conversation_path,
        "conversation_audio_path": conversation_audio_path,
        "conversation_transcript_path": conversation_transcript_path,
        "perception_probe_audio_path": perception_probe_audio_path,
        "perception_probe_raw_events_path": perception_probe_raw_events_path,
    }


async def run_gemini(
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    profile: ModelProfile,
    run_index: int,
    run_id: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    with without_proxy_env():
        client = genai.Client(
            api_key=required_env("GOOGLE_API_KEY"),
            http_options=types.HttpOptions(api_version=args.gemini_live_api_version),
        )
        try:
            config = types.LiveConnectConfig(
                response_modalities=[types.Modality.AUDIO],
                output_audio_transcription=types.AudioTranscriptionConfig(),
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
                system_instruction=gemini_text_content(experiment.system_instruction),
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
                    turn_coverage=types.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
                ),
            )
            async with client.aio.live.connect(model=profile.model, config=config) as session:
                log_stage(case, profile, run_index, "model opening")
                await session.send_realtime_input(text=agent_opening_prompt(experiment))
                opening = await collect_gemini_turn(session, args.timeout_seconds)
                artifact_dir = run_artifact_dir(output_dir, run_id, profile.name, case, run_index)

                async def send_caller_audio(audio_path: Path) -> ProviderTurn:
                    caller_pcm = await asyncio.to_thread(read_audio_pcm, audio_path, GEMINI_SAMPLE_RATE)
                    await session.send_realtime_input(activity_start=types.ActivityStart())
                    await session.send_realtime_input(audio=types.Blob(data=caller_pcm, mime_type=f"audio/pcm;rate={GEMINI_SAMPLE_RATE}"))
                    await session.send_realtime_input(activity_end=types.ActivityEnd())
                    return await collect_gemini_turn(session, args.timeout_seconds)

                if experiment.caller_script:
                    caller_audio_turns, responses, controller_decisions = await run_scripted_conversation_loop(
                        experiment,
                        case,
                        profile,
                        run_index,
                        output_dir,
                        opening,
                        send_caller_audio,
                    )
                else:
                    caller_audio_turns, responses, controller_decisions = await run_conversation_loop(
                        experiment,
                        case,
                        profile,
                        run_index,
                        output_dir,
                        artifact_dir,
                        args,
                        opening,
                        send_caller_audio,
                    )
                probe = None
                if experiment.perception_probe:
                    log_stage(case, profile, run_index, "perception probe")

                    async def ask_probe(text: str) -> ProviderTurn:
                        await session.send_realtime_input(text=text)
                        return await collect_gemini_turn(session, args.timeout_seconds)

                    probe = await run_perception_probe(ask_probe, experiment.perception_probe, PERCEPTION_PROBE_MAX_ATTEMPTS)
        finally:
            client.close()
    log_stage(case, profile, run_index, "saving artifacts")
    artifacts = write_artifacts(output_dir, artifact_dir, experiment, case, profile, run_index, run_id, opening, caller_audio_turns, responses, controller_decisions, probe)
    return result_row(experiment, case, profile, run_index, run_id, opening, responses, artifacts, probe)


async def run_qwen(
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    profile: ModelProfile,
    run_index: int,
    run_id: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    if profile.websocket_url is None:
        raise ValueError(f"Profile {profile.name} is missing websocket_url")
    agent = QwenRealtimeVoiceAgent(
        player_id="agent",
        recipient_id="caller",
        instructions=experiment.system_instruction,
        voice=args.qwen_voice,
        model=profile.model,
        websocket_url=profile.websocket_url,
        input_audio_transcription_model=QWEN_INPUT_TRANSCRIPTION_MODEL,
    )
    await agent.connect(api_key=required_env("DASHSCOPE_API_KEY"))
    try:
        log_stage(case, profile, run_index, "model opening")
        opening = await agent.reply_to_text(agent_opening_prompt(experiment))
        artifact_dir = run_artifact_dir(output_dir, run_id, profile.name, case, run_index)

        async def send_caller_audio(audio_path: Path) -> ProviderTurn:
            caller_pcm = await asyncio.to_thread(read_audio_pcm, audio_path, QWEN_SAMPLE_RATE)
            return await agent.reply_to_audio(caller_pcm)

        if experiment.caller_script:
            caller_audio_turns, responses, controller_decisions = await run_scripted_conversation_loop(
                experiment,
                case,
                profile,
                run_index,
                output_dir,
                opening,
                send_caller_audio,
            )
        else:
            caller_audio_turns, responses, controller_decisions = await run_conversation_loop(
                experiment,
                case,
                profile,
                run_index,
                output_dir,
                artifact_dir,
                args,
                opening,
                send_caller_audio,
            )
        probe = None
        if experiment.perception_probe:
            log_stage(case, profile, run_index, "perception probe")
            probe = await run_perception_probe(agent.reply_to_text, experiment.perception_probe, PERCEPTION_PROBE_MAX_ATTEMPTS)
    finally:
        await agent.close()
    log_stage(case, profile, run_index, "saving artifacts")
    artifacts = write_artifacts(output_dir, artifact_dir, experiment, case, profile, run_index, run_id, opening, caller_audio_turns, responses, controller_decisions, probe)
    return result_row(experiment, case, profile, run_index, run_id, opening, responses, artifacts, probe)


def generate_controller_decision(
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    turn_number: int,
    opening: ProviderTurn,
    caller_audio_turns: list[dict],
    responses: list[ProviderTurn],
    model: str,
) -> ControllerDecision:
    prompt = controller_prompt_payload(experiment, case, turn_number, opening, caller_audio_turns, responses, model)
    with without_proxy_env():
        client = OpenAI(api_key=required_env("OPENAI_API_KEY"), timeout=OPENAI_CLIENT_TIMEOUT)
        response = client.responses.create(
            model=prompt["model"],
            input=prompt["input"],
            text=prompt["text"],
        )
    parsed = json.loads(response.output_text)
    return parse_controller_decision(turn_number, parsed, prompt)


def controller_prompt_payload(
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    turn_number: int,
    opening: ProviderTurn,
    caller_audio_turns: list[dict],
    responses: list[ProviderTurn],
    model: str,
) -> dict:
    return {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": experiment.controller_system_prompt,
            },
            {
                "role": "user",
                "content": controller_prompt(experiment, case, turn_number, opening, caller_audio_turns, responses),
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "multiturn_controller",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "decision": {"type": "string", "enum": ["continue", "stop"]},
                        "reason": {"type": "string"},
                        "next_caller_text": {"type": "string"},
                    },
                    "required": ["decision", "reason", "next_caller_text"],
                },
                "strict": True,
            }
        },
    }


def controller_prompt(
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    turn_number: int,
    opening: ProviderTurn,
    caller_audio_turns: list[dict],
    responses: list[ProviderTurn],
) -> str:
    return experiment.controller_user_prompt_template.format(
        scenario_name=case.scenario_name,
        hidden_state=case.hidden_state,
        persona=case.persona,
        caller_turn_index=turn_number,
        max_caller_turns=experiment.max_caller_turns,
        history_format=experiment.history_format,
        history_so_far=json.dumps(conversation_history(opening, caller_audio_turns, responses), indent=2),
        last_agent_response=responses[-1].transcript,
    )


def parse_controller_decision(turn_number: int, parsed: dict, prompt: dict) -> ControllerDecision:
    decision = parsed["decision"]
    reason = parsed["reason"].strip()
    next_caller_text = parsed["next_caller_text"].strip()
    if not reason:
        raise ValueError("Controller decision reason must be non-empty")
    if decision == "continue" and not next_caller_text:
        raise ValueError("Controller decision continue requires non-empty next_caller_text")
    if decision == "stop" and next_caller_text:
        raise ValueError("Controller decision stop requires empty next_caller_text")
    return ControllerDecision(
        turn_number=turn_number,
        decision=decision,
        reason=reason,
        next_caller_text=next_caller_text,
        prompt=prompt,
        stopped_by="controller",
    )


def forced_stop_decision(turn_number: int, max_caller_turns: int) -> ControllerDecision:
    return ControllerDecision(
        turn_number=turn_number,
        decision="stop",
        reason=f"Reached max_caller_turns={max_caller_turns}",
        next_caller_text="",
        prompt={},
        stopped_by="max_caller_turns",
        forced=True,
    )


def agent_empty_decision(turn_number: int) -> ControllerDecision:
    return ControllerDecision(
        turn_number=turn_number,
        decision="stop",
        reason="agent_empty",
        next_caller_text="",
        prompt={},
        stopped_by="agent_empty",
        forced=True,
    )


def conversation_history(opening: ProviderTurn, caller_audio_turns: list[dict], responses: list[ProviderTurn]) -> list[dict]:
    history = [
        {
            "role": "assistant",
            "turn_index": 0,
            "text": opening.transcript,
        }
    ]
    for index, caller_audio in enumerate(caller_audio_turns):
        turn_number = caller_audio["turn_number"]
        text = caller_audio["generated_caller_text"]
        if not text:
            text = f"[caller audio stimulus: {caller_audio['caller_audio']}]"
        history.append({"role": "user", "turn_index": turn_number, "text": text})
        if index < len(responses):
            history.append({"role": "assistant", "turn_index": turn_number, "text": responses[index].transcript})
    return history


def synthesize_caller_turn(output_dir: Path, artifact_dir: Path, case: MultiturnCase, text: str, turn_number: int, args: argparse.Namespace) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    text_path = artifact_dir / f"generated_turn_{turn_number}.txt"
    text_path.write_text(text, encoding="utf-8")
    audio_path = artifact_dir / f"generated_turn_{turn_number}.mp3"
    voice_id = args.elevenlabs_voice_id or case.caller_voice_id
    if voice_id is None:
        raise RuntimeError(f"Missing ElevenLabs voice id for case {case.case_id}. Add voice_id to experiment.json or pass --elevenlabs-voice-id.")
    payload = {
        "text": text,
        "model_id": args.elevenlabs_model_id,
        "voice_settings": {
            "stability": args.elevenlabs_stability,
            "similarity_boost": args.elevenlabs_similarity_boost,
            "style": args.elevenlabs_style,
            "use_speaker_boost": args.elevenlabs_use_speaker_boost,
        },
    }
    tts_request_path = artifact_dir / f"caller_tts_request_{turn_number}.json"
    tts_request = {
        "voice_id": voice_id,
        "output_format": args.elevenlabs_output_format,
        "payload": payload,
    }
    tts_request_path.write_text(json.dumps(tts_request, indent=2) + "\n", encoding="utf-8")
    request = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format={args.elevenlabs_output_format}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "xi-api-key": required_env("ELEVENLABS_API_KEY"),
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    with without_proxy_env():
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                audio = response.read()
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ElevenLabs TTS failed: HTTP {error.code}: {body}") from error
    if not audio:
        raise RuntimeError("ElevenLabs TTS returned empty audio")
    audio_path.write_bytes(audio)
    return {
        "audio_path": str(audio_path.relative_to(output_dir)),
        "text_path": str(text_path.relative_to(output_dir)),
        "tts_request_path": str(tts_request_path.relative_to(output_dir)),
    }


async def collect_gemini_turn(session, timeout_seconds: float) -> ProviderTurn:
    async def collect() -> ProviderTurn:
        audio_chunks = []
        transcript_parts = []
        raw_events = []
        async for message in session.receive():
            raw_events.append(gemini_event_to_jsonable(message))
            if message.server_content:
                transcription = message.server_content.output_transcription
                if transcription and transcription.text:
                    transcript_parts.append(transcription.text)
                model_turn = message.server_content.model_turn
                if model_turn and model_turn.parts:
                    for part in model_turn.parts:
                        inline_data = part.inline_data
                        if inline_data and inline_data.data:
                            audio_chunks.append(inline_data.data)
                if message.server_content.turn_complete:
                    return ProviderTurn(
                        speaker_id="agent",
                        recipient_id="caller",
                        modality="voice",
                        content=b"".join(audio_chunks),
                        transcript="".join(transcript_parts),
                        media_bytes=b"".join(audio_chunks),
                        response_id="",
                        response_phase="",
                        raw_events=raw_events,
                    )
        raise RuntimeError("Gemini Live session ended before turn_complete")

    return await asyncio.wait_for(collect(), timeout=timeout_seconds)


def probe_is_valid(transcript: str | None) -> bool:
    if not transcript:
        return False
    if PROBE_REFUSAL_RE.search(transcript):
        return False
    return bool(PROBE_VALID_RE.search(transcript))


async def run_perception_probe(ask, text: str, max_attempts: int) -> ProviderTurn:
    last_turn = None
    for _ in range(max_attempts):
        turn = await ask(text)
        last_turn = turn
        if probe_is_valid(turn.transcript):
            return turn
    return last_turn


def result_row(
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    profile: ModelProfile,
    run_index: int,
    run_id: str,
    opening: ProviderTurn,
    responses: list[ProviderTurn],
    artifacts: dict,
    probe: ProviderTurn | None = None,
) -> dict:
    final_response = responses[-1] if responses else ProviderTurn(
        speaker_id="agent",
        recipient_id="caller",
        modality="voice",
        content=b"",
        transcript="",
        media_bytes=b"",
        response_id="",
        response_phase="",
        raw_events=[],
    )
    final_turn = artifacts["response_turns"][-1] if artifacts["response_turns"] else {}
    return {
        "run_id": run_id,
        "run_index": run_index,
        "experiment_id": experiment.experiment_id,
        "case_id": case.case_id,
        "scenario_name": case.scenario_name,
        "task": TASK,
        "condition": case.condition,
        "hidden_state": case.hidden_state,
        "persona": case.persona,
        "max_caller_turns": experiment.max_caller_turns,
        "caller_voice_name": case.caller_voice_name,
        "caller_voice_id": case.caller_voice_id,
        "model_profile": profile.name,
        "model": profile.model,
        "agent_opening_prompt": agent_opening_prompt(experiment),
        "agent_opening_transcript": opening.transcript,
        "artifact_dir": artifacts["artifact_dir"],
        "agent_opening_audio_path": artifacts["agent_opening_audio_path"],
        "caller_audio": case.caller_audio,
        "agent_response_transcript": final_response.transcript,
        "agent_response_audio_path": final_turn["agent_response_audio_path"] if final_turn else "",
        "response_turns": artifacts["response_turns"],
        "generated_caller_text": artifacts["generated_caller_text"],
        "generated_caller_audio_path": artifacts["generated_caller_audio_path"],
        "generated_caller_tts_request_path": artifacts["generated_caller_tts_request_path"],
        "controller_decisions": artifacts["controller_decisions"],
        "stopped_by": artifacts["stopped_by"],
        "agent_opening_raw_events_path": artifacts["agent_opening_raw_events_path"],
        "agent_response_raw_events_path": final_turn["agent_response_raw_events_path"] if final_turn else "",
        "conversation_path": artifacts["conversation_path"],
        "conversation_audio_path": artifacts["conversation_audio_path"],
        "conversation_transcript_path": artifacts["conversation_transcript_path"],
        "perception_probe_prompt": experiment.perception_probe or "",
        "perception_probe_transcript": probe.transcript if probe is not None else "",
        "perception_probe_refused": probe is not None and not probe_is_valid(probe.transcript),
        "perception_probe_audio_path": artifacts["perception_probe_audio_path"],
        "perception_probe_raw_events_path": artifacts["perception_probe_raw_events_path"],
        "error": None,
    }


def agent_opening_prompt(experiment: MultiturnExperiment) -> str:
    return experiment.opening_prompt_template.format(opening_line=experiment.opening_line)


def gemini_text_content(text: str):
    return types.Content(parts=[types.Part.from_text(text=text)])


def experiment_output_dir(output_root: Path, experiment: MultiturnExperiment) -> Path:
    return output_root / TASK / experiment.experiment_id


def run_artifact_dir(output_dir: Path, run_id: str, model_profile: str, case: MultiturnCase, run_index: int) -> Path:
    return output_dir / "artifacts" / f"{run_id}_{model_profile}_{case.case_id}_{run_index:03d}"


def resolve_caller_audio_path(output_dir: Path, stimuli_dir: Path, caller_audio_turn: dict) -> Path:
    path = Path(caller_audio_turn["caller_audio"])
    if path.is_absolute():
        return path
    if caller_audio_turn["caller_audio_source"] == "artifact":
        return output_dir / path
    if caller_audio_turn["caller_audio_source"] == "stimulus":
        return stimuli_dir / path
    raise ValueError(f"Unsupported caller audio source: {caller_audio_turn['caller_audio_source']}")


def write_turn_audio(output_dir: Path, artifact_dir: Path, phase: str, turn: ProviderTurn) -> str:
    path = artifact_dir / f"{phase}.wav"
    write_wav(path, require_pcm(turn.media_bytes, phase), OUTPUT_SAMPLE_RATE)
    return str(path.relative_to(output_dir))


def write_conversation_audio(
    output_dir: Path,
    artifact_dir: Path,
    stimuli_dir: Path,
    opening: ProviderTurn,
    response_turns: list[dict],
    responses: list[ProviderTurn],
) -> str:
    path = artifact_dir / "conversation.wav"
    silence = b"\x00" * int(OUTPUT_SAMPLE_RATE * 0.7) * SAMPLE_WIDTH_BYTES
    parts = [require_pcm(opening.media_bytes, "agent_opening")]
    for response_turn, response in zip(response_turns, responses):
        parts.append(silence)
        caller_audio_path = Path(response_turn["caller_audio"])
        if not caller_audio_path.is_absolute():
            if response_turn["caller_audio_source"] == "artifact":
                caller_audio_path = output_dir / caller_audio_path
            elif response_turn["caller_audio_source"] == "stimulus":
                caller_audio_path = stimuli_dir / caller_audio_path
            else:
                raise ValueError(f"Unsupported caller audio source: {response_turn['caller_audio_source']}")
        parts.append(read_audio_pcm(caller_audio_path, OUTPUT_SAMPLE_RATE))
        parts.append(silence)
        parts.append(require_pcm(response.media_bytes, f"agent_response_{response_turn['turn_number']}"))
    pcm = b"".join(parts)
    write_wav(path, pcm, OUTPUT_SAMPLE_RATE)
    return str(path.relative_to(output_dir))


def write_conversation_transcript(
    output_dir: Path,
    artifact_dir: Path,
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    opening: ProviderTurn,
    response_turns: list[dict],
    responses: list[ProviderTurn],
    probe: ProviderTurn | None = None,
) -> str:
    path = artifact_dir / "transcript.txt"
    text = (
        f"case_id: {case.case_id}\n"
        f"scenario_name: {case.scenario_name}\n"
        f"condition: {case.condition}\n"
        f"hidden_state: {case.hidden_state}\n"
        f"persona: {case.persona}\n"
        f"max_caller_turns: {experiment.max_caller_turns}\n"
        f"caller_voice_name: {case.caller_voice_name}\n"
        f"caller_audio: {', '.join(case.caller_audio)}\n\n"
        f"USER (agent opening prompt): {agent_opening_prompt(experiment)}\n\n"
        f"ASSISTANT (agent opening): {opening.transcript}\n\n"
    )
    for response_turn, response in zip(response_turns, responses):
        turn_number = response_turn["turn_number"]
        text += (
            f"USER (caller audio {turn_number}): {response_turn['caller_audio']}\n"
            f"USER (caller text {turn_number}): {response_turn['generated_caller_text']}\n\n"
            f"ASSISTANT (agent response {turn_number}): {response.transcript}\n\n"
            f"CONTROLLER ({turn_number}): {json.dumps(response_turn['controller_decision'], ensure_ascii=True)}\n\n"
        )
    if probe is not None:
        text += (
            f"USER (perception probe): {experiment.perception_probe}\n\n"
            f"ASSISTANT (perception probe): {probe.transcript}\n\n"
        )
    path.write_text(text, encoding="utf-8")
    return str(path.relative_to(output_dir))


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(SAMPLE_WIDTH_BYTES)
        file.setframerate(sample_rate)
        file.writeframes(pcm)


def require_pcm(pcm: bytes | None, label: str) -> bytes:
    if pcm is None or not pcm:
        raise ValueError(f"Missing audio bytes for {label}")
    if len(pcm) % SAMPLE_WIDTH_BYTES != 0:
        raise ValueError(f"Audio byte length is not 16-bit aligned for {label}")
    return pcm


def write_raw_events(output_dir: Path, artifact_dir: Path, phase: str, events: list) -> str:
    path = artifact_dir / f"raw_events_{phase}.json"
    path.write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(output_dir))


def write_controller_prompt(output_dir: Path, artifact_dir: Path, turn_number: int, prompt: dict) -> str:
    path = artifact_dir / f"controller_prompt_{turn_number}.json"
    path.write_text(json.dumps(prompt, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(output_dir))


def controller_decision_to_jsonable(decision: ControllerDecision) -> dict:
    return {
        "turn_number": decision.turn_number,
        "decision": decision.decision,
        "reason": decision.reason,
        "next_caller_text": decision.next_caller_text,
        "stopped_by": decision.stopped_by,
        "forced": decision.forced,
    }


def stopped_by(controller_decisions: list[ControllerDecision]) -> str:
    if not controller_decisions:
        return ""
    return controller_decisions[-1].stopped_by


def stopped_by_value_from_response_turns(response_turns: list[dict]) -> str:
    if not response_turns:
        return ""
    return response_turns[-1]["controller_decision"]["stopped_by"]


def latest_generated_caller_turn(response_turns: list[dict]) -> dict | None:
    generated = [turn for turn in response_turns if turn["caller_audio_source"] == "artifact"]
    if not generated:
        return None
    return generated[-1]


def write_conversation(
    output_dir: Path,
    artifact_dir: Path,
    experiment: MultiturnExperiment,
    case: MultiturnCase,
    profile: ModelProfile,
    run_index: int,
    run_id: str,
    opening: ProviderTurn,
    responses: list[ProviderTurn],
    opening_audio_path: str,
    opening_raw_events_path: str,
    response_turns: list[dict],
    conversation_audio_path: str,
    conversation_transcript_path: str,
    probe: ProviderTurn | None = None,
    perception_probe_audio_path: str = "",
    perception_probe_raw_events_path: str = "",
) -> str:
    path = artifact_dir / "conversation.json"
    conversation = {
        "run_id": run_id,
        "run_index": run_index,
        "experiment_id": experiment.experiment_id,
        "case_id": case.case_id,
        "scenario_name": case.scenario_name,
        "task": TASK,
        "condition": case.condition,
        "max_caller_turns": experiment.max_caller_turns,
        "hidden_state": case.hidden_state,
        "persona": case.persona,
        "caller_voice_name": case.caller_voice_name,
        "caller_voice_id": case.caller_voice_id,
        "model_profile": profile.name,
        "model": profile.model,
        "system_instruction": experiment.system_instruction,
        "conversation_audio_path": conversation_audio_path,
        "conversation_transcript_path": conversation_transcript_path,
        "controller_decisions": [response_turn["controller_decision"] for response_turn in response_turns],
        "stopped_by": stopped_by_value_from_response_turns(response_turns),
        "turns": [
            {
                "role": "user",
                "type": "text",
                "purpose": "agent_opening_prompt",
                "text": agent_opening_prompt(experiment),
            },
            {
                "role": "assistant",
                "type": "audio_response",
                "purpose": "agent_opening",
                "transcript": opening.transcript,
                "audio_path": opening_audio_path,
                "raw_events_path": opening_raw_events_path,
            },
        ],
    }
    for response_turn in response_turns:
        conversation["turns"].append(
            {
                "role": "user",
                "type": "audio",
                "purpose": f"caller_response_{response_turn['turn_number']}",
                "audio_path": response_turn["caller_audio"],
                "audio_source": response_turn["caller_audio_source"],
                "generated_text": response_turn["generated_caller_text"],
                "generated_text_path": response_turn["generated_caller_text_path"],
                "tts_request_path": response_turn["generated_caller_tts_request_path"],
            }
        )
        conversation["turns"].append(
            {
                "role": "assistant",
                "type": "audio_response",
                "purpose": f"agent_response_{response_turn['turn_number']}",
                "transcript": response_turn["agent_response_transcript"],
                "audio_path": response_turn["agent_response_audio_path"],
                "raw_events_path": response_turn["agent_response_raw_events_path"],
                "controller_decision": response_turn["controller_decision"],
                "controller_prompt_path": response_turn["controller_prompt_path"],
            }
        )
    if probe is not None:
        conversation["turns"].append(
            {
                "role": "user",
                "type": "text",
                "purpose": "perception_probe_prompt",
                "text": experiment.perception_probe,
            }
        )
        conversation["turns"].append(
            {
                "role": "assistant",
                "type": "audio_response",
                "purpose": "perception_probe",
                "transcript": probe.transcript,
                "audio_path": perception_probe_audio_path,
                "raw_events_path": perception_probe_raw_events_path,
            }
        )
    path.write_text(json.dumps(conversation, indent=2) + "\n", encoding="utf-8")
    return str(path.relative_to(output_dir))


def gemini_event_to_jsonable(message) -> dict:
    try:
        return json.loads(message.model_dump_json())
    except Exception:
        return {"event": str(message)}


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


def append_result(output_dir: Path, row: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
