"""
Audio-native labeler using an OpenAI audio-capable model. Requires
OPENAI_API_KEY in env. Verify the current audio-input model name before
running -- check OpenAI's docs, this changes frequently.
"""
import base64
import json
import os
from openai import OpenAI

from schema import LabelerResult, Segment
from prompt import SYSTEM_PROMPT, USER_PROMPT

MODEL = os.getenv("OPENAI_AUDIO_MODEL", "gpt-audio")


def label_call(audio_path: str, call_id: str) -> LabelerResult:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "input_audio",
                 "input_audio": {"data": audio_b64, "format": "wav"}},
            ]},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:500] if raw else "<empty response>"
        raise ValueError(f"OpenAI audio returned non-JSON output: {preview}") from exc
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
    return LabelerResult(labeler="gpt_audio", call_id=call_id,
                          segments=segments, raw_response=raw,
                          model=MODEL, usage=usage)
