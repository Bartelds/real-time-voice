from dataclasses import dataclass
from datetime import UTC, datetime


QWEN_REALTIME_WEBSOCKET_URL = "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"


MODEL_PROFILES = {
    "openai_realtime": {
        "provider": "openai_realtime",
        "model": "gpt-realtime-2",
    },
    "gemini_live": {
        "provider": "gemini_live",
        "model": "gemini-3.1-flash-live-preview",
    },
    "gemini_frontier": {
        "provider": "gemini_frontier",
        "model": "gemini-3.1-pro-preview",
    },
    "qwen_realtime": {
        "provider": "qwen_realtime",
        "model": "qwen3.5-omni-plus-realtime",
        "websocket_url": QWEN_REALTIME_WEBSOCKET_URL,
    },
    "qwen_flash_realtime": {
        "provider": "qwen_realtime",
        "model": "qwen3.5-omni-flash-realtime",
        "websocket_url": QWEN_REALTIME_WEBSOCKET_URL,
    },
}

REALTIME_MODEL_PROFILE_NAMES = ("openai_realtime", "gemini_live", "qwen_realtime", "qwen_flash_realtime")


@dataclass(frozen=True)
class ModelProfile:
    name: str
    provider: str
    model: str
    websocket_url: str | None = None


def model_profile(name: str) -> ModelProfile:
    row = MODEL_PROFILES[name]
    return ModelProfile(
        name=name,
        provider=row["provider"],
        model=row["model"],
        websocket_url=row["websocket_url"] if "websocket_url" in row else None,
    )


def make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
