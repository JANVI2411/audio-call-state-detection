"""
Transcript-only labeler: ASR (local, no network needed for this half) +
a text-only LLM classification pass. Deliberately has NO access to
prosody, tone, or hold music -- it can only reason from words and
silence gaps. This gives you a labeler whose failure mode is genuinely
different from the two audio-native models: it will miss things like
"flat scripted human" (no acoustic cue available) but it will NOT
hallucinate off crosstalk the way an audio-native model can, so its
disagreements are informative rather than redundant.

Requires faster-whisper locally and an LLM API key for the classification
pass (can reuse either provider, or a cheaper text-only model).
"""
import json
import os
from faster_whisper import WhisperModel
from openai import OpenAI

from schema import LabelerResult, Segment

ASR_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small.en")
CLASSIFY_MODEL = os.getenv("OPENAI_CLASSIFY_MODEL", "gpt-5-mini")

CLASSIFY_PROMPT = """You are given a timestamped transcript of one side of \
a phone call (the far-end party only; the AI agent's own turns are marked \
[AGENT]). Silence gaps longer than 1.5s are marked [SILENCE Xs].

Using ONLY the words and silence pattern (you cannot hear tone of voice or \
music), label each transcript segment as one of: ivr, human, survey, hold, \
unknown. Long silences with no words are usually "hold" or "unknown" -- \
you cannot distinguish hold music from dead air from text alone, so default \
to "unknown" rather than guessing "hold" unless the words around it say \
"please hold" or similar.

Output a JSON array of {{start, end, label, confidence, evidence, human_id}}.
Transcript:
{transcript}
"""


def _format_transcript(whisper_segments) -> str:
    lines = []
    prev_end = 0.0
    for seg in whisper_segments:
        gap = seg.start - prev_end
        if gap > 1.5:
            lines.append(f"[SILENCE {gap:.1f}s]")
        lines.append(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text.strip()}")
        prev_end = seg.end
    return "\n".join(lines)


def label_call(audio_path: str, call_id: str) -> LabelerResult:
    asr = WhisperModel(ASR_MODEL_SIZE, device="cpu", compute_type="int8")
    whisper_segments, _ = asr.transcribe(audio_path, vad_filter=True)
    transcript = _format_transcript(list(whisper_segments))

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=CLASSIFY_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user",
                   "content": CLASSIFY_PROMPT.format(transcript=transcript)}],
    )
    raw = (response.choices[0].message.content or "").strip()
    parsed = json.loads(raw)
    items = parsed if isinstance(parsed, list) else parsed.get("segments", [])
    segments = [
        Segment(
            start=float(s["start"]), end=float(s["end"]), label=s["label"],
            confidence=float(s.get("confidence", 0.5)),
            evidence=s.get("evidence", ""), human_id=s.get("human_id"),
        )
        for s in items
    ]
    usage = response.usage.model_dump() if response.usage else {}
    return LabelerResult(labeler="asr_llm", call_id=call_id,
                          segments=segments, raw_response=raw,
                          model=CLASSIFY_MODEL, usage=usage)
