"""
Audio-native labeler using Gemini. Requires GEMINI_API_KEY in env.
"""
import base64
import json
import os
from google import genai

from schema import LabelerResult, Segment
from prompt import SYSTEM_PROMPT, USER_PROMPT

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
RESPONSE_FORMAT = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "number"},
            "end": {"type": "number"},
            "label": {
                "type": "string",
                "enum": ["ivr", "human", "survey", "hold", "unknown"],
            },
            "confidence": {"type": "number"},
            "evidence": {"type": "string"},
            "human_id": {"type": "string"},
        },
        "required": ["start", "end", "label", "confidence", "evidence"],
    },
}


def label_call(audio_path: str, call_id: str) -> LabelerResult:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    interaction = client.interactions.create(
        model=MODEL,
        system_instruction=SYSTEM_PROMPT,
        input=[
            {"type": "text", "text": USER_PROMPT},
            {"type": "audio", "data": audio_b64, "mime_type": "audio/wav"},
        ],
        generation_config={"temperature": 0.0},
        response_format=RESPONSE_FORMAT,
    )

    raw = (interaction.output_text or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:500] if raw else "<empty response>"
        raise ValueError(f"Gemini returned non-JSON output: {preview}") from exc
    segments = [
        Segment(
            start=float(s["start"]), end=float(s["end"]), label=s["label"],
            confidence=float(s.get("confidence", 0.5)),
            evidence=s.get("evidence", ""), human_id=s.get("human_id"),
        )
        for s in parsed
    ]
    usage = getattr(interaction, "usage_metadata", None)
    if usage is None:
        usage = getattr(interaction, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif usage is None:
        usage = {}

    return LabelerResult(labeler="gemini", call_id=call_id,
                          segments=segments, raw_response=raw,
                          model=MODEL, usage=usage)
