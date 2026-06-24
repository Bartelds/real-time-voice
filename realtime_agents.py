import base64
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

import websockets


Modality = Literal["voice", "text"]
JsonPrimitive = str | int | float | bool | None
JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class VoiceProfile:
    voice: str
    input_format: str = "audio/pcm"
    output_format: str = "audio/pcm"
    sample_rate: int = 24000
    silence_gap_seconds: float = 0.55
    max_sentences: int | None = None


@dataclass(frozen=True)
class RealtimeConfig:
    model: str = "gpt-realtime-2"
    websocket_url: str = "wss://api.openai.com/v1/realtime"
    output_modalities: list[str] = field(default_factory=lambda: ["audio"])
    turn_detection: dict[str, JsonValue] | None = None
    reasoning_effort: str = "medium"
    max_response_output_tokens: int | str | None = None


@dataclass
class ProviderTurn:
    speaker_id: str
    recipient_id: str
    modality: Modality
    content: str | bytes
    transcript: str
    media_bytes: bytes | None
    response_id: str | None
    response_phase: str | None
    raw_events: list[dict[str, Any]]


SUPPORTED_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "marin",
    "sage",
    "shimmer",
    "verse",
}

SUPPORTED_AUDIO_FORMATS = {
    "audio/pcm",
    "audio/pcmu",
    "audio/pcma",
}


def validate_voice_profile(profile: VoiceProfile) -> None:
    if profile.voice not in SUPPORTED_VOICES:
        raise ValueError(f"Unsupported voice: {profile.voice}")
    if profile.input_format not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"Unsupported input format: {profile.input_format}")
    if profile.output_format not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"Unsupported output format: {profile.output_format}")
    if profile.input_format == "audio/pcm" and profile.sample_rate != 24000:
        raise ValueError("PCM input audio must use a 24000 Hz sample rate")
    if profile.output_format == "audio/pcm" and profile.sample_rate != 24000:
        raise ValueError("PCM output audio must use a 24000 Hz sample rate")
    if profile.silence_gap_seconds < 0:
        raise ValueError("silence_gap_seconds must be non-negative")
    if profile.max_sentences is not None and profile.max_sentences < 1:
        raise ValueError("max_sentences must be positive")


class RealtimeVoiceAgent:
    def __init__(
        self,
        player_id: str,
        recipient_id: str,
        instructions: str,
        voice_profile: VoiceProfile,
        realtime_config: RealtimeConfig,
    ) -> None:
        validate_voice_profile(voice_profile)
        self.player_id = player_id
        self.recipient_id = recipient_id
        self.instructions = instructions
        self.voice_profile = voice_profile
        self.realtime_config = realtime_config
        self.ws: Any = None

    async def connect(self, api_key: str) -> None:
        self.ws = await websockets.connect(
            f"{self.realtime_config.websocket_url}?model={self.realtime_config.model}",
            additional_headers=[
                ("Authorization", f"Bearer {api_key}"),
            ],
            max_size=None,
        )
        await self._wait_for("session.created")
        await self._send(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": self.realtime_config.model,
                    "instructions": self.instructions,
                    "output_modalities": self.realtime_config.output_modalities,
                    "reasoning": {
                        "effort": self.realtime_config.reasoning_effort,
                    },
                    "audio": {
                        "input": {
                            "format": self._audio_format(self.voice_profile.input_format, include_rate=True),
                            "turn_detection": self.realtime_config.turn_detection,
                        },
                        "output": {
                            "format": self._audio_format(self.voice_profile.output_format, include_rate=True),
                            "voice": self.voice_profile.voice,
                        },
                    },
                },
            }
        )
        await self._wait_for("session.updated")

    async def close(self) -> None:
        await self.ws.close()

    async def opening_turn(self) -> ProviderTurn:
        await self._send(
            {
                "type": "response.create",
                "response": self._response_payload(),
            }
        )
        return await self._collect_turn()

    async def reply_to_audio(self, audio: bytes, text: str | None = None) -> ProviderTurn:
        content = [
            {
                "type": "input_audio",
                "audio": base64.b64encode(audio).decode("ascii"),
            }
        ]
        if text is not None:
            content.append(
                {
                    "type": "input_text",
                    "text": text,
                }
            )
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": content,
                },
            }
        )
        await self._send(
            {
                "type": "response.create",
                "response": self._response_payload(),
            }
        )
        return await self._collect_turn()

    async def reply_to_text(self, text: str) -> ProviderTurn:
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text,
                        }
                    ],
                },
            }
        )
        await self._send(
            {
                "type": "response.create",
                "response": self._response_payload(),
            }
        )
        return await self._collect_turn()

    async def reply(self, turn: ProviderTurn) -> ProviderTurn:
        if turn.media_bytes is None:
            raise ValueError("Realtime voice replies require audio media bytes")
        return await self.reply_to_audio(turn.media_bytes)

    async def _collect_turn(self) -> ProviderTurn:
        chunks = []
        transcript = ""
        raw_events = []
        response_id = ""
        while True:
            event = json.loads(await self.ws.recv())
            raw_events.append(event)
            event_type = event["type"]
            if event_type == "response.output_audio.delta":
                chunks.append(base64.b64decode(event["delta"]))
            elif event_type == "response.output_audio_transcript.done":
                transcript = event["transcript"]
            elif event_type == "response.done":
                response = event["response"]
                response_id = response["id"]
                response_phase = response["status"]
                if response["status"] == "completed" and response["output"]:
                    response_phase = response["output"][0]["phase"]
                audio = b"".join(chunks)
                return ProviderTurn(
                    speaker_id=self.player_id,
                    recipient_id=self.recipient_id,
                    modality="voice",
                    content=audio,
                    transcript=transcript,
                    media_bytes=audio,
                    response_id=response_id,
                    response_phase=response_phase,
                    raw_events=raw_events,
                )
            elif event_type == "error":
                raise RuntimeError(json.dumps(event, indent=2))

    async def _wait_for(self, expected_type: str) -> None:
        while True:
            event = json.loads(await self.ws.recv())
            if event["type"] == expected_type:
                return
            if event["type"] == "error":
                raise RuntimeError(json.dumps(event, indent=2))

    async def _send(self, event: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(event))

    def _audio_format(self, audio_format: str, include_rate: bool) -> dict[str, Any]:
        if audio_format == "audio/pcm":
            payload = {"type": audio_format}
            if include_rate:
                payload["rate"] = self.voice_profile.sample_rate
            return payload
        return {
            "type": audio_format,
        }

    def _response_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "output_modalities": self.realtime_config.output_modalities,
        }
        if self.realtime_config.max_response_output_tokens is not None:
            payload["max_output_tokens"] = self.realtime_config.max_response_output_tokens
        return payload


# DashScope rejects websocket frames over 256 KiB (close code 1009), so audio is
# streamed in 1-second PCM chunks (16 kHz mono 16-bit = 32000 B -> ~43 KB base64).
AUDIO_APPEND_CHUNK_BYTES = 32000


class QwenRealtimeVoiceAgent:
    def __init__(
        self,
        player_id: str,
        recipient_id: str,
        instructions: str,
        voice: str,
        model: str,
        websocket_url: str,
        output_modalities: tuple[str, ...] = ("text", "audio"),
        input_audio_transcription_model: str | None = None,
    ) -> None:
        self.player_id = player_id
        self.recipient_id = recipient_id
        self.instructions = instructions
        self.voice = voice
        self.model = model
        self.websocket_url = websocket_url
        self.output_modalities = list(output_modalities)
        self.input_audio_transcription_model = input_audio_transcription_model
        self.ws: Any = None

    async def connect(self, api_key: str) -> None:
        self.ws = await websockets.connect(
            f"{self.websocket_url}?model={self.model}",
            additional_headers=[
                ("Authorization", f"Bearer {api_key}"),
            ],
            max_size=None,
        )
        await self._wait_for("session.created")
        session: dict[str, Any] = {
            "modalities": self.output_modalities,
            "voice": self.voice,
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "instructions": self.instructions,
            "turn_detection": None,
        }
        if self.input_audio_transcription_model is not None:
            session["input_audio_transcription"] = {"model": self.input_audio_transcription_model}
        await self._send({"type": "session.update", "session": session})

    async def close(self) -> None:
        await self.ws.close()

    async def request_response(self, user_text: str = ".") -> ProviderTurn:
        return await self.reply_to_text(user_text)

    async def update_instructions(self, instructions: str) -> None:
        self.instructions = instructions
        await self._send({"type": "session.update", "session": {"instructions": instructions}})

    async def reply_to_audio(self, audio: bytes, text: str | None = None) -> ProviderTurn:
        for start in range(0, len(audio), AUDIO_APPEND_CHUNK_BYTES):
            await self._send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio[start : start + AUDIO_APPEND_CHUNK_BYTES]).decode("ascii"),
                }
            )
        await self._send({"type": "input_audio_buffer.commit"})
        await self._send({"type": "response.create"})
        return await self._collect_turn()

    async def reply_to_text(self, text: str) -> ProviderTurn:
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        await self._send({"type": "response.create"})
        return await self._collect_turn()

    async def _collect_turn(self) -> ProviderTurn:
        chunks = []
        transcript = ""
        raw_events = []
        while True:
            event = json.loads(await self.ws.recv())
            raw_events.append(event)
            event_type = event["type"]
            if event_type == "response.audio.delta":
                chunks.append(base64.b64decode(event["delta"]))
            elif event_type == "response.audio_transcript.done":
                transcript = event["transcript"]
            elif event_type == "response.text.done":
                if not transcript:
                    transcript = event["text"]
            elif event_type == "response.done":
                response = event.get("response", {})
                audio = b"".join(chunks)
                return ProviderTurn(
                    speaker_id=self.player_id,
                    recipient_id=self.recipient_id,
                    modality="voice",
                    content=audio,
                    transcript=transcript,
                    media_bytes=audio,
                    response_id=response.get("id", ""),
                    response_phase=response.get("status", ""),
                    raw_events=raw_events,
                )
            elif event_type == "error":
                raise RuntimeError(json.dumps(event, indent=2))

    async def _wait_for(self, expected_type: str) -> None:
        while True:
            event = json.loads(await self.ws.recv())
            if event["type"] == expected_type:
                return
            if event["type"] == "error":
                raise RuntimeError(json.dumps(event, indent=2))

    async def _send(self, event: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(event))
