"""
Turns segment-level gold JSONL into a HuggingFace dataset where each ROW
is one trimmed audio clip -- this is what gets you the same "play the
row, see the label" browsing experience as a flat classification
dataset, while keeping your project's actual segment-level schema
(start/end/label/confidence/human_id/agreement/source) as columns.

Run: python build_hf_dataset.py --gold-dir ./gold --audio-dir ./calls \
        --repo-id YOUR_USERNAME/call-party-segments --push
"""
import argparse
import glob
import json
import os

import soundfile as sf
from datasets import Dataset, Audio, Features, Value


def load_gold(gold_dir):
    rows = []
    for path in glob.glob(os.path.join(gold_dir, "*.gold.jsonl")):
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def trim_and_write(audio_dir, row, clips_dir, pad=0.5):
    src = os.path.join(audio_dir, f"{row['call_id']}.beeped.wav")
    data, sr = sf.read(src)
    s = max(0, int((row["start"] - pad) * sr))
    e = min(len(data), int((row["end"] + pad) * sr))
    clip_path = os.path.join(
        clips_dir, f"{row['call_id']}_{row['start']:.2f}_{row['end']:.2f}.wav")
    sf.write(clip_path, data[s:e], sr)
    return clip_path


def build(gold_dir, audio_dir, clips_dir, split_name):
    os.makedirs(clips_dir, exist_ok=True)
    rows = load_gold(gold_dir)

    records = []
    for row in rows:
        clip_path = trim_and_write(audio_dir, row, clips_dir)
        # NOTE: no PHI leaves this record -- only timing/labels/confidence,
        # never transcript text (see project's no-PHI-leakage requirement)
        records.append({
            "audio": clip_path,
            "call_id": row["call_id"],
            "start": row["start"],
            "end": row["end"],
            "label": row["label"],
            "agreement": row.get("agreement"),
            "reviewed": "final_label" in row,
            "final_label": row.get("final_label", row["label"]),
            "human_id": row.get("human_id"),
            "split_tag": split_name,  # normal / hard / edge_case -- set upstream
        })

    features = Features({
        "audio": Audio(sampling_rate=16000),
        "call_id": Value("string"),
        "start": Value("float32"),
        "end": Value("float32"),
        "label": Value("string"),
        "agreement": Value("float32"),
        "reviewed": Value("bool"),
        "final_label": Value("string"),
        "human_id": Value("string"),
        "split_tag": Value("string"),
    })
    return Dataset.from_list(records, features=features)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", default="./gold")
    ap.add_argument("--audio-dir", default="./calls")
    ap.add_argument("--clips-dir", default="./clips")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--split-name", default="normal")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    ds = build(args.gold_dir, args.audio_dir, args.clips_dir, args.split_name)
    print(ds)

    if args.push:
        # Requires: huggingface-cli login  (or HF_TOKEN env var)
        ds.push_to_hub(args.repo_id, split=args.split_name)
        print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")
    else:
        ds.save_to_disk("./hf_dataset_local")
        print("Saved locally to ./hf_dataset_local (rerun with --push to upload)")
