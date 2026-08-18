"""
Minimal human review UI. Loads a review queue (segments flagged by
vote.needs_human_review), plays the audio clip for each, shows what
each labeler proposed, and lets a reviewer accept or correct.

Run: python review_app.py --queue review_queue.jsonl --audio-dir ./calls
Writes decisions to reviewed.jsonl as you go (safe to stop/resume).
"""
import argparse
import json
import os

import gradio as gr
import soundfile as sf

from schema import LABELS


def load_queue(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def trim_clip(audio_path, start, end, pad=1.0):
    data, sr = sf.read(audio_path)
    s = max(0, int((start - pad) * sr))
    e = min(len(data), int((end + pad) * sr))
    return sr, data[s:e]


def build_app(queue_path, audio_dir, out_path):
    queue = load_queue(queue_path)
    state = {"idx": 0}

    def render(idx):
        if idx >= len(queue):
            return None, "Queue complete.", "", "", gr.update(visible=False)
        item = queue[idx]
        audio_path = os.path.join(audio_dir, f"{item['call_id']}.beeped.wav")
        sr, clip = trim_clip(audio_path, item["start"], item["end"])
        model_votes = "\n".join(
            f"{k}: {v}" for k, v in item.get("model_labels", {}).items()
        )
        info = (f"call {item['call_id']}  [{item['start']:.1f}s - {item['end']:.1f}s]  "
                f"reason: {item['reason']}\n\nmodel votes:\n{model_votes}")
        return (sr, clip), info, item.get("label", LABELS[0]), "", gr.update(visible=True)

    def submit(label, human_id, note):
        item = queue[state["idx"]]
        item["final_label"] = label
        item["human_id"] = human_id or None
        item["reviewer_note"] = note
        with open(out_path, "a") as f:
            f.write(json.dumps(item) + "\n")
        state["idx"] += 1
        return render(state["idx"])

    with gr.Blocks() as demo:
        gr.Markdown("## Segment review")
        audio = gr.Audio(label="Clip (padded 1s each side)")
        info = gr.Textbox(label="Context", lines=6, interactive=False)
        label = gr.Radio(LABELS, label="Correct label")
        human_id = gr.Textbox(label="human_id (only if label = human)")
        note = gr.Textbox(label="Reviewer note (optional)")
        btn = gr.Button("Accept and next")

        demo.load(lambda: render(0), outputs=[audio, info, label, human_id, btn])
        btn.click(submit, inputs=[label, human_id, note],
                  outputs=[audio, info, label, human_id, btn])

    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="review_queue.jsonl")
    ap.add_argument("--audio-dir", default="./calls")
    ap.add_argument("--out", default="reviewed.jsonl")
    args = ap.parse_args()
    build_app(args.queue, args.audio_dir, args.out).launch()
