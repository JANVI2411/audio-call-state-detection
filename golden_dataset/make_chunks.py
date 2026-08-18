"""
Split long call recordings into fixed-length chunks for labeling.

Why chunking is needed
----------------------
The labelers send the whole audio file to an API in one request. A ten
minute call is a big request, the models drift on long audio, and one bad
response costs you the whole call. Cutting the call into 5 minute pieces
means each request is small, a failure only costs one piece, and pieces can
be labeled in any order.

What this writes
----------------
chunks/<call>_c00.wav      the audio piece itself
chunks/manifest.json       where each piece came from, so the labels can be
                           put back on the original call's clock later

Format note: pieces are written as plain 16-bit WAV. The source files are
mu-law (the compressed format phone networks use), and the labeling APIs
expect ordinary WAV, so the conversion happens here rather than being left
as a surprise at upload time.

The tail piece
--------------
A 601.8 second call does not divide evenly into 300 second pieces; the
leftover is 1.8 seconds, which is too short to label usefully. Any leftover
shorter than --min-tail is merged into the piece before it, so the last
piece is allowed to run long rather than leaving a scrap behind.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import List

import soundfile as sf


@dataclass
class Chunk:
    chunk_id: str        # unique name, used as the call-id when labeling
    source_file: str     # which recording it was cut from
    source_call: str     # short name of that recording
    index: int           # 0, 1, 2, ... within that recording
    offset_s: float      # where this piece starts in the original call
    duration_s: float    # how long this piece is
    path: str            # where the piece was written


def plan_chunks(total_s: float, chunk_s: float, min_tail_s: float) -> List[tuple]:
    """Return a list of (start, end) pairs covering the whole recording."""
    spans = []
    start = 0.0
    while start < total_s:
        end = min(start + chunk_s, total_s)
        spans.append((start, end))
        start = end
    # Merge a too-short final piece into the one before it.
    if len(spans) > 1 and (spans[-1][1] - spans[-1][0]) < min_tail_s:
        last_start, last_end = spans.pop()
        prev_start, _ = spans.pop()
        spans.append((prev_start, last_end))
    return spans


def cut_file(audio_path: str, out_dir: str, chunk_s: float,
             min_tail_s: float) -> List[Chunk]:
    info = sf.info(audio_path)
    total_s = info.frames / info.samplerate
    # Short name: everything before the first dot of the file name.
    call_name = os.path.basename(audio_path).split(".")[0]
    short = call_name[:8]

    spans = plan_chunks(total_s, chunk_s, min_tail_s)
    out: List[Chunk] = []

    for i, (start_s, end_s) in enumerate(spans):
        start_frame = int(round(start_s * info.samplerate))
        stop_frame = int(round(end_s * info.samplerate))
        data, sr = sf.read(audio_path, start=start_frame, stop=stop_frame)

        chunk_id = f"{short}_c{i:02d}"
        path = os.path.join(out_dir, f"{chunk_id}.wav")
        # PCM_16 is the plain uncompressed form every API accepts.
        sf.write(path, data, sr, subtype="PCM_16")

        out.append(Chunk(
            chunk_id=chunk_id,
            source_file=os.path.abspath(audio_path),
            source_call=call_name,
            index=i,
            offset_s=round(start_s, 3),
            duration_s=round(end_s - start_s, 3),
            path=os.path.abspath(path),
        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cut call recordings into chunks.")
    ap.add_argument("--input-dir", default="./input")
    ap.add_argument("--out-dir", default="./chunks")
    ap.add_argument("--chunk-minutes", type=float, default=5.0)
    ap.add_argument("--min-tail-seconds", type=float, default=60.0,
                    help="Leftover shorter than this is merged into the "
                         "previous chunk instead of standing alone.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    chunk_s = args.chunk_minutes * 60.0

    wavs = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".wav")
    )
    if not wavs:
        raise SystemExit(f"No .wav files found in {args.input_dir}")

    all_chunks: List[Chunk] = []
    for wav in wavs:
        info = sf.info(wav)
        dur = info.frames / info.samplerate
        print(f"\n{os.path.basename(wav)}")
        print(f"  length {dur:.1f}s ({dur/60:.2f} min), "
              f"{info.samplerate} Hz, {info.channels} channel(s), {info.subtype}")
        chunks = cut_file(wav, args.out_dir, chunk_s, args.min_tail_seconds)
        for c in chunks:
            size_mb = os.path.getsize(c.path) / 1e6
            print(f"  -> {c.chunk_id}  {c.offset_s:7.1f}s to "
                  f"{c.offset_s + c.duration_s:7.1f}s  "
                  f"({c.duration_s:6.1f}s, {size_mb:.1f} MB)")
        all_chunks.extend(chunks)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2)

    total_min = sum(c.duration_s for c in all_chunks) / 60.0
    print(f"\n{len(all_chunks)} chunks, {total_min:.1f} minutes of audio total")
    print(f"manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
