# Golden dataset pipeline for call-party detection

Bootstraps segment-level gold labels (ivr / human / survey / hold / unknown)
using three independent labelers, votes on a frame grid, routes disagreement
and a spot-check sample to human review, and publishes the result as a
playable HuggingFace dataset (one row = one trimmed clip).

**This cannot run inside the environment that generated it** — no network,
no audio, no API keys there. Run it on your own machine.

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
huggingface-cli login
```

Place redacted recordings in `./calls/<call_id>.beeped.wav`.

## 1. Label + vote + route, per call

```bash
for f in calls/*.beeped.wav; do
  call_id=$(basename "$f" .beeped.wav)
  python run_pipeline.py --audio "$f" --call-id "$call_id" --out-dir ./gold
done
```

Produces `./gold/<call_id>.gold.jsonl` (full segment timeline, every
segment tagged with vote agreement) and `./gold/review_queue.jsonl`
(segments needing a human look: true splits, mandatory; agreed segments,
spot-checked at 12%/30% rates — see `vote.needs_human_review`).

## 2. Human review

```bash
python review_app.py --queue ./gold/review_queue.jsonl --audio-dir ./calls
```

Opens a local Gradio app: plays each flagged clip, shows what each model
voted, lets you accept or correct. Writes `reviewed.jsonl`; safe to stop
and resume.

Merge `reviewed.jsonl` back into the gold files (final_label overrides
label) before the next step — a two-line jq/pandas join, intentionally
left as a manual step so you can spot-check the merge before publishing.

## 3. Tag edge cases and split

Before building the HF dataset, tag calls matching your corpus's known
hard cases (human-sounding IVR, flat scripted human, hold interrupting
human speech, multi-transfer calls) with `split_tag: hard` or
`edge_case`; everything else defaults to `normal`. This is a deliberate
manual/scripted step — don't rely solely on emergent disagreement rate to
populate the hard set (see design note below).

## 4. Build and publish

```bash
python build_hf_dataset.py \
  --gold-dir ./gold --audio-dir ./calls \
  --repo-id YOUR_USERNAME/call-party-segments \
  --split-name normal --push

python build_hf_dataset.py \
  --gold-dir ./gold_hard --audio-dir ./calls \
  --repo-id YOUR_USERNAME/call-party-segments \
  --split-name hard --push
```

Each row is a trimmed clip (±0.5s padding) with `audio`, `label`,
`agreement`, `human_id`, and provenance columns — plays inline in the HF
dataset viewer exactly like the flat-classification reference dataset,
but preserves segment-level structure your eval actually needs.

## Design notes (why it's built this way)

- **Frame-grid voting, not raw segment comparison.** Segment boundaries
  never line up between labelers; "agreement" is undefined without a
  common grid. See `vote.py`.
- **Three labelers, not two.** Two audio-native models can be confidently
  and identically wrong on the same crosstalk — agreement between them
  isn't truth. The third labeler (transcript-only) has a genuinely
  different failure mode: no access to prosody, so it can't hallucinate
  off it.
- **Mandatory spot-check on full-agreement segments.** Disagreement-only
  review will never catch a blind spot all three labelers share (e.g.
  every model calling a flat scripted human "ivr"). The spot-check rate
  is a knob (`vote.needs_human_review`), not a nice-to-have.
- **No PHI in the published dataset.** Only timing, labels, confidence,
  and provenance are written — never transcript text — per the project's
  no-PHI-leakage requirement. Audio clips are trimmed from the already
  `.beeped.wav` sources.
