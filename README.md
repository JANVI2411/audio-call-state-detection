# Call state detection for an outbound voice agent

An AI agent calls insurance companies. Before it can say anything useful it
has to know one thing: **who is on the other end right now?** An automated
menu, a live person, or hold music — and whether it just got transferred.

## Why this is two systems, not one

Large audio models solve the accuracy half of this problem well. Gemini Flash
and GPT-audio both identify IVR and human reliably on real payer calls — they
hear the whole recording, reason over it, and get it right.

They cannot be used during the call. Each request sends the entire audio and
waits for a complete answer, which means:

- you need the call to be **over** before you can start
- every request is a **network round trip**, seconds of it
- you **pay per call**, forever
- the audio **leaves your infrastructure** — these are health-care calls

So accuracy is available but latency is not, and a voice agent needs the
answer while the person is still talking.

That splits the work in two:

```
   REAL CALLS
       |
       +---------------------------+
       v                           v
  golden_dataset              live_call_pred
                                          
  big models, offline         small local model, live
  slow, costs money           fast, free, no network
  accurate                    still catching up
       |                           ^
       +------ teaches ------------+
```

**The big models are the teacher. The live pipeline is the student.** We use
the expensive, accurate, slow path to produce labelled data, and spend that
data making a cheap, fast path good enough to run during the call.

This repo is both halves.

## The main design decision

Most obvious approach: one classifier picking `ivr` / `human` / `hold` /
`transfer`. We deliberately did not do that.

**A transfer is not a sound.** While a transfer is happening you are hearing
hold music. A four-way classifier is forced into a choice reality does not
offer — call it `hold` and the transfer is never reported, call it
`transfer` and you lose the fact that music is playing.

So they are split:

- **states** — `ivr`, `human`, `hold`, `other`: what the audio *is*
- **events** — transfer started/ended, speaker changed: what is *happening*

Both are true at once. The state stays `hold` while a transfer is separately
in progress. A transfer is confirmed from converging evidence, none of which
is sufficient alone:

```
rep A talking
   |  "let me transfer you"        <- language
hold music                          <- state change
   |  new voice                     <- speaker change
rep B talking                       <- confirmed
```

An announcement is a *prediction*, and predictions can be wrong. If nothing
follows one, we report a **failed** transfer. A system that only reported
successes would look flawless while silently dropping every broken handoff.

---

## Part 1 — `golden_dataset`: making the answer key

Runs offline, after the call. Costs money (two audio models plus one text
model per chunk).

```
input/ long recording
   |
   v  make_chunks.py        cut into 5-minute pieces, mu-law -> plain WAV
chunks/
   |
   v  label_chunks.py       pick pieces, look up their offsets
   |
   +-> gemini_labeler   -+
   +-> gpt_labeler       +-> three independent opinions (cached on disk)
   +-> asr_llm_labeler  -+
   |
   v  repair.py            fix broken timestamps
   v  vote.py              merge the three into one answer
   |
   +-> gold/<chunk>.gold.jsonl        the answer key
   +-> gold/review_queue.<chunk>      what a human should check
   +-> gold/metrics/<chunk>           cost and timing
```

**Three labelers, not two.** Two audio-native models can be confidently and
identically wrong on the same crosstalk; agreement between them is not truth.
The third works from a transcript only, so it fails differently.

### How the three get merged

Their segment boundaries never line up, so "agreement" is undefined until
they share a timeline. We cut at the **union of every boundary any labeler
proposed**:

```
gemini    |------- ivr -------|--- human ---|--- ivr ---|
gpt_aud   |--- ivr ---|------ human ------|---- ivr ----|
asr_llm   |- ivr -|- human -|------- ivr -------|-------|
          ^       ^   ^     ^     ^        ^    ^       ^
          every boundary becomes a cut point
```

Inside any resulting piece **nobody changes their mind**, so each labeler has
exactly one vote there and the count is exact — nothing to round.

Four rules that came out of real failures:

| Rule | The failure it fixes |
|---|---|
| A labeler that covered nothing **abstains** | Filling gaps with "unknown" let silence outvote the labelers that did the work — one chunk came out 70% unknown |
| Cut at real boundaries, not a fixed grid | Rounding start down and end up grew every segment, so neighbours overlapped and the later one won. Boundaries drifted **+56 ms on average, always late** |
| Ties are flagged and sent to a human | `Counter.most_common` returned whichever label was inserted first — the tie went to whoever happened to be first in the list |
| Merge only on same label **and** same speaker | Merging on label alone collapsed thirteen turns between two people into one `human` block, hiding a transfer |

### Run it

```bash
cd golden_dataset
cp .env.example .env          # add your keys
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python make_chunks.py
.venv/bin/python label_chunks.py --first-only        # or --chunk-id <id>
.venv/bin/python review_app.py --queue gold/review_queue.<chunk>.jsonl
```

Finished labelers are cached, so re-running after a failure costs nothing.

---

## Part 2 — `live_call_pred`: deciding during the call

Runs locally. No network, no cost. Uses only audio it has already heard —
never looks ahead.

```
    audio in ---> RING BUFFER (last 6s, fixed size)
                        |
      +-----------+-----+-----+------------+
      v           v           v            v
   SOUND       WORDS      SPEAKER      CARRIER
   ~4 ms      ~2000 ms     ~0 ms        ~0 ms
   speech?    speech-to-   same         keypad
   music?     text, then   person       tones,
   tone?      match IVR /  as           new call
   pitch?     hold phrases before?      legs
      |           |           |            |
      +-----------+-----+-----+------------+
                        v
              26 named signals + 32 embedding numbers
              x3 (now / recent average / the change) = 174
                        v
              WEIGH EVIDENCE  -> ivr .61 human .22 hold .12
                        v
              SMOOTH (HMM)    -> states persist; one odd moment cannot flip
                        v
              COMMIT          -> switching needs more proof than staying
                        v
         +--------------+---------------+
         v                              v
      STATE                          EVENTS
```

**No layer is allowed to be the final answer.** "Music is present" never
directly means `hold`, because announcements are routinely played *over* hold
music. Evidence goes up; the decision comes down.

### Run it

```bash
cd live_call_pred
python3 -u scripts/run_spec.py \
  --wav <chunk.wav> --segment silence \
  --asr faster_whisper --gold auto --out out/spec
```

Prints each decision as it is made and flushes it to disk immediately, so a
crash keeps everything up to that point. `--gold auto` shows the answer key
beside each prediction with a running accuracy.

Outputs per call: `timeline.jsonl` (segments with `start, end, label,
sub_label, human_id, confidence, evidence`), `events.jsonl`, `summary.json`,
and `hops.jsonl` (every individual decision, for debugging).

```bash
python3 tests/run_all.py     # 148 tests, no network or model download
```

---

## Where it actually stands

Measured on 10 minutes of real payer audio, against the three-model answer
key. Nothing here is from synthetic data.

| | 69f8c04e | 69f3a1e4 |
|---|---|---|
| accuracy | **0.404** | **0.317** |

Started at 0.255 / 0.267. The jump came from **35 phrase patterns written
against transcripts the system actually produced** — no model change, no
training. IVR correctly identified went 7 -> 21 on the first chunk, and
`ivr -> human` errors fell from 22 to 7.

**Latency**, silence segmentation, per decision:

| | fixed 2 s | silence |
|---|---|---|
| wall time for 300 s | 254 s | **122 s** |
| worst decision | 10.9 s | **2.7 s** |
| over its own budget | 49% | **23%** |
| backlog at call end | 9.9 s | **0.0 s** |

Everything except speech recognition runs in about 5 ms with almost no
variance. One component costs roughly 400x the rest combined.

### What does not work yet

- **~36% accuracy.** IVR precision is high (~0.93) but recall is low: it is
  right when it commits and stays quiet too often.
- **`human` is the leftover bucket.** Words matching no pattern default there.
- **Speaker identity is assigned during hold and IVR.** Transfers are detected
  by noticing the speaker changed, so a phantom speaker registered on hold
  music makes a genuine handoff look like the same person continuing.
- **23% of decisions still miss their latency budget.**

### What we would do next

**1. Distil the big models into a small one.** This is the main line of work.
Gemini Flash and GPT-audio already produce the right answers; they are just
too slow and too expensive to call live. Every labelled call is a worked
example of what the fast path should have said. With enough of them, finetune
a small model — a compact audio encoder, or a small language model over the
transcript plus acoustic features — that runs locally in milliseconds.

The point is not to beat the big models. It is to get close enough while
being **hundreds of times faster and free per call**. The pipeline is already
built to accept this: the state head sits behind one interface with three
implementations, so a trained model drops in without touching anything
around it.

**2. Move speech recognition off the critical path.** It costs roughly 400x
everything else combined and is the only thing breaking the latency budget.
Run it alongside the decision loop, use the most recent words available, and
keep deciding on sound when they are stale. That turns a hard stall into
graceful degradation.

**3. Keep growing the labelled set.** 10 minutes is enough to expose bugs and
nowhere near enough to train on. 20–50 calls is the target. The labelling
pipeline exists precisely so this is cheap to do.

**4. Smaller fixes** — require positive evidence for `human` rather than
letting it be the leftover bucket; stop registering speaker identity during
hold and IVR.

### On the synthetic numbers

There is a synthetic corpus and a trained head that scores 98.3% on held-out
calls. That is reported in small print on purpose: the audio has no line
noise, no codec artifacts and no real speaker variation, and calls within a
scenario share templates, so it only demonstrates within-template
generalisation. The default model is a **hand-written weight table with no
training in it at all** — every real-call number above came from that.

---

## Data and keys

No audio, transcripts or keys are in this repo. Both are gitignored:

- `.env` holds live API keys — copy `.env.example` and add your own
- recordings are real health-care calls; the audio is redacted but the
  transcripts derived from it are not

Supply your own recordings in `golden_dataset/input/` to run the pipeline.
