# Real-Time Voice AI Hears but Does Not Listen

## Abstract

> Speech conveys information through both words and vocal delivery. We evaluate four leading
> production realtime voice systems—OpenAI's GPT Realtime 2, Google's Gemini 3.1 Flash Live, and
> Alibaba's Qwen3.5 Omni Plus and Omni Flash—on tasks where the words and the delivery patterns both
> convey meaningful information. Across three consequential scenarios, all four systems act on the
> words rather than the voice. They end calls with crying callers who insist nothing is wrong,
> approve wire transfers authorized in frightened voices, and enroll callers whose agreement is
> clearly sarcastic. Surprisingly, this is often not a failure of perception. When asked directly,
> three of the four systems reliably identify the distress, fear, or sarcasm they later ignore when
> making decisions. We observe a similar pattern when these realtime voice systems estimate accent
> and age, as their responses frequently follow the biases of the words rather than the acoustic
> properties of the speaker. We term this disconnect between perception and action the
> *emotional intelligence gap* of voice AI. Prompting systems to explicitly attend to vocal delivery
> improves performance only partially and inconsistently. Our findings show that current realtime
> voice AI systems often behave as if speech had been reduced to a transcript, suggesting that they
> should be used with caution in settings where the tone and emotion of delivery convey important
> information.

This repository contains the code, prompts, and stimuli for the paper. Audio recordings are released
separately as a dataset (see [Data](#data)).

## Systems

- OpenAI GPT Realtime 2 (`gpt-realtime-2`)
- Google Gemini 3.1 Flash Live (`gemini-3.1-flash-live-preview`)
- Alibaba Qwen3.5 Omni Plus Realtime (`qwen3.5-omni-plus-realtime`)
- Alibaba Qwen3.5 Omni Flash Realtime (`qwen3.5-omni-flash-realtime`)
- Text-only baseline: Gemini 3.1 Pro (`gemini-3.1-pro-preview`)
- Caller turns: GPT-5.5 (`gpt-5.5`)
- Speech synthesis: ElevenLabs (`eleven_v3`)

## Setup

Requires Python >= 3.12 and `ffmpeg` on the PATH.

```bash
pip install -r requirements.txt
```

Create a `.env` in the repo root with the keys you need:

```
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
DASHSCOPE_API_KEY=...
ELEVENLABS_API_KEY=...
```

## Collect model responses

Each command calls the live model APIs and writes its per-run outputs to `runs/`.

```bash
# multi-turn scenarios (append _attend or _override to the id for the instruction variants)
python run_multiturn_voice.py --experiment-id welfare_callback
python run_multiturn_voice.py --experiment-id wire_fraud
python run_multiturn_voice.py --experiment-id volunteer_recruitment

# single-turn diagnostics
python run.py --task accent_perception
python run.py --task age

# delivery diagnostics (distress / fear / sarcasm)
python run.py --cases cases_discrimination_distress.json --prompts prompts_discrimination.json
python run.py --cases cases_discrimination_fear.json --prompts prompts_discrimination.json
python run.py --cases cases_discrimination_sarcasm.json --prompts prompts_discrimination.json

# text-only baseline for the delivery diagnostics (words-only floor, Gemini 3.1 Pro)
python -m discrimination.text_floor_distress
python -m discrimination.text_floor_fear
python -m discrimination.text_floor_sarcasm
```

Pass `--help` for options.

## Score and summarize responses

Each command reads the per-run outputs from `runs/` and writes a summary CSV.

```bash
python analyze_accent.py
python analyze_age.py
python -m discrimination.analyze_distress
python -m discrimination.analyze_fear
python -m discrimination.analyze_sarcasm
```

## Data

The stimulus audio is in this repo under `stimuli/`. The audio recordings are released as a dataset
on Hugging Face: <https://huggingface.co/datasets/bartelds/real-time-voice>.

<!--
## Citation

TODO: add citation (BibTeX) once the paper is public.
-->

## License

Code is released under the MIT License (see `LICENSE`). All speech was synthesized with
[ElevenLabs](https://elevenlabs.io) (`eleven_v3`) and is made available **solely to document this research.**
Please do not reuse it to train, evaluate, benchmark, or otherwise build machine-learning or AI
systems, or redistribute it as a standalone audio collection.
