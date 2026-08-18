# callstate — real-time call-state tracking for an outbound voice agent

Tracks, causally and in real time, what is on the far end of an outbound
call: **IVR**, **HUMAN**, **HOLD**, or **OTHER** (ringback, dead air, tones) —
and, separately, whether a **transfer** is under way and whether it succeeded.

The central design commitment, and the reason this is not a four-class chunk
classifier:

> **IVR / HUMAN / HOLD / OTHER are persistent states. TRANSFER is an event** —
> a transition inferred from state history, speaker change, language and
> telephony evidence.

A flat `softmax(HUMAN, IVR, HOLD, TRANSFER)` forces a choice reality does not
offer. During a transfer the audio *is* hold music, so the flat model must
either call it HOLD (and never report the transfer) or call it TRANSFER (and
lose the fact that hold music is playing). Here they coexist: the state stays
`hold` while `transfer_in_progress` is independently true. That single
decision removes a whole class of errors before any modelling starts.

---

## Quick start

No network, no model downloads, no GPU. Runs on numpy + scipy alone.

```bash
cd gpt_idea

# 1. the whole test suite (144 tests, stdlib unittest, ~35 s)
python3 tests/run_all.py

# 2. generate a labelled synthetic corpus (24 calls, 4 scenarios)
python3 scripts/make_synthetic.py --per-scenario 6

# 3. run one call end to end and score it
python3 scripts/run_call.py \
    --wav data/synthetic/transfer_0.wav --call-id transfer_0 \
    --script data/synthetic/transfer_0.script.json \
    --telephony data/synthetic/transfer_0.telephony.jsonl \
    --asr scripted --agent-channel 0 --evaluate

# 4. score the whole corpus
python3 scripts/evaluate.py

# 5. train the fusion head and compare against the built-in prior weights
python3 scripts/train_fusion.py
python3 scripts/evaluate.py --model-path models/fusion_head.npz \
                            --holdout-from models/fusion_head.meta.json
```

Real ASR and the pretrained encoders are optional extras:

```bash
pip install faster-whisper                 # real transcription
pip install torch transformers             # --audio-encoder wavlm
pip install sentence-transformers          # --text-encoder minilm

python3 scripts/run_call.py --wav /path/to/call.wav --call-id real_call \
    --asr faster_whisper --text-encoder minilm --evaluate
```

---

## Architecture

```
                        remote audio (RTP / websocket / WAV)
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
              agent leg (ours)                  counterparty leg
              — privileged, we                  — the thing being
                know when WE talk                  classified
                    │                                 │
                    │                    ┌────────────┴────────────┐
                    │                    │   RingBuffer  6 s       │
                    │                    │   causal, no look-ahead │
                    │                    └────────────┬────────────┘
                    │                                 │
                    │        ┌────────────┬───────────┼───────────┬─────────────┐
                    │        ▼            ▼           ▼           ▼             ▼
                    │   VAD + modulation  DTMF     log-mel /    streaming    speaker
                    │   music / tone /   Goertzel   WavLM        ASR         embedding
                    │   pitch / energy              embedding   (whisper)   + change
                    │        │            │           │           │             │
                    │        │            │           │           ▼             │
                    │        │            │           │      lexicon:           │
                    │        │            │           │      IVR / hold /       │
                    │        │            │           │      transfer phrases   │
                    │        │            │           │           │             │
        telephony   └────────┴────────────┴───────────┴───────────┴─────────────┘
        SIP / DTMF                          │
        leg changes ───────────────────────►│
                                            ▼
                              FEATURE FUSION  (featurizer.py)
                        24 named scalars + projected embeddings,
                        each as [current | context mean | delta]
                                            │
                                            ▼
                              STATE MODEL  (model.py)
                     prior weights (no data) | logistic (trained) | GRU
                                            │
                                     emission probabilities
                                            │
                                            ▼
                              HMM FORWARD FILTER  (hmm.py)
                          causal; suppresses single-hop flicker
                                            │
                                            ▼
                              STATE TRACKER  (state_tracker.py)
                          commits a state; tracks dwell time
                                            │
                        ┌───────────────────┴───────────────────┐
                        ▼                                       ▼
                  STATE OUTPUT                            EVENT OUTPUT
                  ivr / human / hold / other               transfer_start
                  + posterior + confidence                 transfer_end (completed|failed)
                                                           speaker_changed
                                                           human_joined
                                                           ivr_exit
                                            │
                                            ▼
                    timeline.jsonl · events.jsonl · segments.jsonl
                    latency.jsonl · summary.json · logs/<call>.log.jsonl
```

Streaming geometry: a **6 s** causal context window, an inference **hop of
0.5 s**, acoustic decisions read from the trailing **2.5 s**, lexical scoring
over the trailing **4 s**, and ASR on its own **1 s** cadence over a **3 s**
window. Every one of those numbers is justified below and set in
[config.py](callstate/config.py).

---

## Results

All numbers below were produced by the commands in this README. Two things to
read them against: they are measured on a **synthetic** corpus, and the
caveats section says plainly what that does and does not establish.

### States, on 8 held-out calls (845 hops)

The held-out calls are excluded from training and stratified so each scenario
appears in both halves.

| | accuracy | macro-F1 | ECE | excess state changes/hr |
|---|---|---|---|---|
| built-in prior weights (**no training data**) | 0.804 | 0.772 | 0.185 | 120 |
| trained logistic head | **0.963** | **0.957** | **0.028** | **0** |

Trained head, per state:

| state | precision | recall | F1 | support |
|---|---|---|---|---|
| ivr | 0.970 | 0.956 | 0.963 | 274 |
| human | 0.991 | 0.961 | 0.976 | 334 |
| hold | 0.950 | 0.962 | 0.956 | 157 |
| other | 0.870 | 1.000 | 0.930 | 80 |

Boundary recall: 0.42 within ±1 s, 1.00 within ±2 s. Stability: 26 predicted
state changes against 26 in ground truth — no flicker.

### Transfers, on 4 transfer calls

| | tp | fp | fn | P | R | F1 |
|---|---|---|---|---|---|---|
| transfer_start | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| transfer_end | 4 | 0 | 0 | 1.00 | 1.00 | 1.00 |

Completed-vs-failed outcome accuracy **1.00**, including the case that
separates a real detector from a naive one: an announcement, then hold, then
*the same representative* returns. Acoustically that is identical to a
successful transfer — only speaker identity distinguishes them, and it is
reported as `failed`.

### Latency

Measured per hop on a laptop CPU, single core, no GPU:

| stage | mean | p95 |
|---|---|---|
| acoustic front-end | 3.5 ms | 4.0 ms |
| ASR buffering (scripted) | 14.6 ms | 16.1 ms |
| speaker branch | 0.15 ms | 0.32 ms |
| fusion + HMM + events | 0.21 ms | 0.14 ms |
| **total** | **15.6 ms** | **16.7 ms** |

Against a 500 ms hop budget that is a real-time factor of **0.03** — about 30x
headroom. Adding real faster-whisper `small.en` on CPU pushes a hop to roughly
1–2 s, i.e. **over budget**; see the honesty section.

---

## Design decisions, and the measurements behind them

Every threshold in this package was set from a measurement. Where a first
attempt failed, the failure is documented in the code rather than quietly
replaced — a reviewer should be able to see what was tried.

### The modulation spectrum, not autocorrelation, separates speech from music

Hold detection is the load-bearing acoustic problem. The first attempt used
long-lag envelope autocorrelation to find the musical loop. It did not work:
at a 6 s window, a 4 s lag is computed from 2 s of overlap, so it returned
noise, and it scored **speech higher than hold music** — the opposite of the
intent.

What works is the **envelope modulation spectrum**. Speech is amplitude-
modulated at the syllable rate (2.5–8 Hz) with remarkable consistency; music
and tones are not. Measured:

| | syllable band (2.5–8 Hz) | slow band (0.05–1.2 Hz) |
|---|---|---|
| speech (IVR) | 0.92 | 0.02 |
| speech (human) | 0.86 | 0.06 |
| hold music | 0.14 | 0.36 |
| ringback tone | 0.02 | 0.73 |
| silence | 0.07 | 0.02 |
| **prompt over hold music** | **0.48** | **0.20** |

That last row is the case a flat classifier cannot express, and the reason the
front-end emits overlapping multi-label scores instead of a decision.

### Two analysis windows, because two timescales genuinely conflict

Boundary responsiveness wants a *short* decision window — stale audio in the
decision is a lower bound on detection lag. But music detection lives at
0.05–1.2 Hz, and a 2.5 s window has 0.4 Hz resolution, so that band is not
measurable in it at all.

Using one short window for everything drove **hold recall to exactly zero**.
Using one long window for everything put boundaries ~1.5 s late. The fix is to
measure the slow band over the full 6 s window and everything responsive over
the trailing 2.5 s: speech starting is visible immediately, "we are in a music
bed" is allowed to take longer to establish. That matches how the two
phenomena actually behave.

### Speech evidence must not depend on energy contrast alone

An energy VAD needs speech to stand out from the floor, so it fails completely
on a prompt spoken over a continuous hold-music bed — measured voiced fraction
**0.00** while syllable modulation read **0.62**. The speech was plainly
there and the VAD could not see it. Because syllable modulation is an energy
*fraction*, a raised floor does not affect it. The two are averaged, not
chained: chaining lets either one veto, averaging lets either one carry the
evidence.

### Tone detection needs two peaks and the full window

Every North American call-progress tone is a *pair* — ringback 440+480 Hz,
busy 480+620. Scoring only the strongest peak splits the energy and
under-reports them. Two peaks, measured over 6 s windows across 25 seeds:

| | range |
|---|---|
| speech | 0.008 – 0.016 |
| hold music | 0.073 – 0.101 |
| ringback | 0.202 – 0.321 |

The threshold `tone_min_inband_fraction = 0.15` sits in the empty gap. It also
has to be measured over the *long* window: ringback is ~2 s on and ~4 s off,
so a 2.5 s window can land entirely in the silent phase and score zero.

### Speaker identity is what makes a transfer verifiable

`HUMAN_A → hold → HUMAN_B` is a near-conclusive transfer signature, and it is
only visible if B can be told from A. Measured on MFCC embeddings (C0 dropped,
see below): same-speaker cosine **≥ 0.997**, different-speaker **≤ 0.751**.
`speaker_similarity_threshold = 0.86` sits in that gap.

Two failures fixed along the way, both preserved as regression tests:

- **C0 dominated the embedding.** MFCC coefficient 0 is log frame energy, so
  leaving it in made the "voice embedding" mostly a loudness measure and the
  same person at two volumes stopped matching. Dropping it fixes it.
- **A phantom third speaker** appeared at every hold→speech boundary: the
  first window carries the tail of the music, producing an embedding matching
  nobody. Requiring two consecutive agreeing observations before committing a
  new identity removes it.

### The HMM is what makes the output actionable

Taking argmax of a per-window distribution produces flicker — `ivr, ivr,
human, ivr, human` — because individual windows are genuinely ambiguous. A
causal forward filter with transition priors means one odd window cannot move
the call state while several consistent ones can. Only the forward recursion
is used; forward-backward would be more accurate but requires the future,
which does not exist yet in a live call.

The transition priors encode call structure rather than symmetry: `hold→human`
outweighs `hold→ivr` because coming off hold to a representative is the common
case, and `human→ivr` is low because a live person rarely hands you back to a
menu without hold in between.

### A transfer may not resolve on a single hop

Resolving on the first hop of a new state was expensive in two directions. A
single flickered IVR hop inside hold closed the transfer as *completed*; a
return-from-hold hop arriving one hop before the speaker branch committed the
new identity closed it as *failed* against the outgoing speaker. Both are
cases where waiting two seconds costs nothing and guessing costs the whole
event, so a candidate state must settle before it can resolve a transfer.

### ASR runs on its own cadence

Feeding the full 6 s context to a recogniser every 0.5 s hop re-transcribes
each second of audio twelve times. Words do not change once spoken, so a 3 s
window every 1 s cuts the redundancy to 3x, and between ASR runs the state
loop reuses the most recent transcript — exactly what a real streaming
recogniser's partial hypotheses provide.

De-duplication has to work on **content and time**, not time alone. Each
window is decoded independently, so word timings drift by a few hundred
milliseconds between overlapping decodes; a word emitted at 5.90 s reappears
at 6.05 s, clears a strictly-increasing cutoff, and is emitted twice. On the
real call that produced `"for calling BlueCard. Card Eligibility.
Eligibility."` — which then breaks phrase matching, because "thank you for
calling" no longer appears contiguously.

### Language carries the IVR decision

Acoustically a good recorded prompt and a calm representative are similar.
Lexically they are nothing alike. `"for pharmacy benefits, press or say one"`
is unambiguous where the waveform is not. Two lexical corrections worth
noting:

- `"your call is important to us"` is a **hold**-queue script, not a menu
  prompt. Having it in the IVR list made every hold segment with a spoken
  overlay classify as IVR.
- `"your call may be recorded"` is a disclosure and stays with IVR.

Patterns are inspectable and fire identically every run, which is why they are
patterns and not a classifier. Their weakness is paraphrase — they will miss
*"I'm going to put you through to somebody in claims"*. `--text-encoder
minilm` plus the trained head covers that, which is why both the pattern
scores and the text embedding go into the feature vector.

### Three state models, because a system must be useful on day zero

| model | needs | use |
|---|---|---|
| `PriorStateModel` | nothing | hand-specified evidence weights over named scalars; runs everywhere; `explain()` returns per-feature contributions |
| `LogisticStateModel` | a few hundred labelled segments | numpy multinomial logistic; the first thing to fit on real data |
| `GRUStateModel` | torch + real volume | learns temporal structure instead of a hand-built context summary |

None of them is the final answer on its own — all three feed the HMM.

---

## Evaluation methodology

Chunk accuracy alone is close to useless here, so `metrics.py` reports four
families:

- **State quality** — macro-F1, not micro. A call is mostly IVR and hold, so a
  model that never predicts HUMAN can post high accuracy while being useless
  for the one decision the agent actually cares about.
- **Boundary quality** — was the transition within ±0.5 / ±1 / ±2 s, with
  human-detection lag broken out. A model can score well per-chunk and still
  be late on every boundary, which is exactly the failure that hurts a live
  agent.
- **Transfer quality** — precision/recall/F1 with a time tolerance, plus
  completed-vs-failed correctness. Scored separately because transfers are
  rare: they contribute almost nothing to frame metrics while being the
  highest-value thing to get right.
- **Stability and latency** — false state changes per hour, latency
  percentiles, per-branch breakdown.

Three methodological points that matter more than the scores:

**Splits are by call, never by frame.** A 6 s window advancing 0.5 s means
consecutive feature vectors share over 90% of their audio. A random frame
split puts near-identical rows on both sides and reports a number that has
nothing to do with generalisation.

**Zero gold events is reported as "not evaluated", never as 1.00.** An
unstratified draw once produced a holdout with no transfer call in it, making
every transfer metric read a vacuous perfect score. `transfer_metrics` now
returns `None` with a note in that case, and the split is stratified.

**Temperature is fitted on held-out data.** The raw head is overconfident
(ECE ≈ 0.19). One temperature parameter, fitted by minimising held-out NLL,
brings it to ≈ 0.03 without moving the decision boundary — so accuracy is
unchanged and the confidence number becomes usable by a policy that says
"act when >0.8 sure a human is on the line".

---

## Where this is honest about its limits

**The headline numbers are synthetic.** `callstate/simulate.py` generates
source-filter speech that reproduces the cues the front-end keys on — pitch
range, syllable-rate modulation, loop periodicity, tone purity — but not codec
artefacts, line noise, crosstalk, or real speaker variability. They are
evidence the mechanism works end to end, not an accuracy claim about
production traffic. `scripts/evaluate.py` prints this caveat with every
report.

**Synthetic variation had to be forced, and this was caught the hard way.**
With calls that differed only by noise realisation, the trained head scored
**100% on both train and a by-call holdout** — a held-out `transfer_0` was
essentially the same recording as the `transfer_1` it trained on. The
generator now draws voices, durations, prompt selection, gain, noise floor and
hold-loop length per call. Even so, calls within a scenario share script
templates, so the holdout measures *within-template* generalisation. It is not
a substitute for real labelled calls.

**On the real call in this repo, the untrained prior weights do not
transfer.** Run against `voice_agent/input/*.beeped.wav` (601.8 s, 8 kHz
stereo mu-law) with real Whisper ASR, the prior model does not reliably
identify the IVR that ground truth says runs for the first 46 s. The main
reason is instructive: **real IVR prompts are recorded human voices**, so the
narrow-pitch cue that separates synthetic TTS from synthetic conversation
simply does not apply. Language remains discriminative; acoustics largely do
not. The fix is not a better hand-tuned weight — it is to label real calls and
fit the head on them, which is what `scripts/train_fusion.py` exists to do.
This is reported rather than hidden because it is the single most important
thing to know before deploying this.

**Real ASR blows the latency budget.** faster-whisper `small.en` on CPU takes
roughly 1–2 s per hop against a 500 ms budget. Options, in the order I would
try them: a purpose-built streaming telephony ASR (Deepgram, AssemblyAI —
native mu-law/8 kHz, real websocket partials, endpointing tuned for phone
audio), a smaller Whisper with a GPU, or widening the ASR cadence further and
leaning more on acoustics between updates. The architecture already isolates
this: ASR is one branch behind one interface.

**Channel routing must not be guessed if it can be known.** The first version
of `detect_agent_channel` used a per-channel relative threshold, which
normalises away the very difference being measured — on a quiet leg the
percentile collapses onto the noise floor, so line noise counts as "active".
On the real call it therefore labelled the *counterparty's* IVR leg as our own
agent, and the pipeline spent the whole call classifying our agent's channel
and reading Whisper hallucinations (`"You"`, `"Yes. Yes."`) off near-silence.
Fixed with a shared absolute threshold, verified correct on the real call and
on all 24 synthetic calls. But it is still inference: in production, take the
routing from the telephony layer, which knows for certain.

**Not tested here**: mono recordings (the free channel split disappears and
real diarization becomes necessary), overlapping speech / crosstalk, non-
English calls, and the `wavlm` / `minilm` / `ecapa` encoder paths, which are
implemented against documented APIs but exercised only through their
interfaces.

---

## What I would do next, in priority order

1. **Label 20–50 real calls and retrain.** Everything else is secondary. The
   machinery — feature capture, by-call splits, temperature scaling, the whole
   metrics suite — is already built and waiting for the data.
2. **Swap MFCC for ECAPA-TDNN speaker embeddings.** Single biggest accuracy
   lever available; MFCC is a classical baseline that will not hold up across
   codec and channel mismatch. One-function change.
3. **Move to a streaming telephony ASR.** Fixes the latency budget and the
   8 kHz mismatch at the same time.
4. **Snap hop boundaries to VAD silence gaps** so a window never lands
   mid-utterance. Does not change the causal contract.
5. **Model crosstalk explicitly.** Nothing here distinguishes overlapping
   speakers; an energy-ratio overlap detector is the cheap first step.

---

## Repository layout

```
gpt_idea/
├── callstate/
│   ├── types.py             State / Event / Observation — the core contracts
│   ├── config.py            every tunable, with its measurement in the comment
│   ├── engine.py            the streaming loop; one hop, start to finish
│   ├── simulate.py          synthetic call generator + exact ground truth
│   ├── metrics.py           state / boundary / transfer / stability / calibration
│   ├── telephony.py         SIP / DTMF / leg-change event bus
│   ├── io_sinks.py          JSONL + summary writers
│   ├── logging_setup.py     console stream + structured JSONL trace
│   ├── audio/
│   │   ├── codecs.py        G.711 mu-law/A-law + RIFF, no third-party deps
│   │   ├── source.py        frame sources, channel routing, RingBuffer
│   │   └── features.py      VAD, modulation spectrum, tone, DTMF, pitch, log-mel
│   ├── encoders/
│   │   ├── audio_encoder.py log-mel stats | WavLM
│   │   ├── text_encoder.py  hashed n-gram | MiniLM
│   │   └── speaker.py       MFCC | ECAPA, registry, change detection
│   ├── semantics/
│   │   ├── asr.py           streaming buffer + faster-whisper / scripted / null
│   │   └── lexicon.py       IVR / hold / transfer / spontaneous-speech patterns
│   └── fusion/
│       ├── featurizer.py    the feature vector contract
│       ├── model.py         prior | logistic | GRU state heads
│       ├── hmm.py           causal forward filter
│       ├── state_tracker.py commit policy + dwell
│       └── events.py        transfer lifecycle + boundary events
├── scripts/
│   ├── make_synthetic.py    build the labelled corpus
│   ├── run_call.py          run one call, write logs, optionally score it
│   ├── train_fusion.py      fit + temperature-scale the state head
│   └── evaluate.py          corpus-wide aggregate report
├── tests/                   144 tests, stdlib unittest, no network
│   ├── run_all.py           the runner (`--fast` skips the slow e2e module)
│   ├── test_codecs.py       G.711 + RIFF, incl. the real call file
│   ├── test_causality.py    ring buffer + engine cannot see the future
│   ├── test_features.py     every acoustic separation the fusion layer needs
│   ├── test_speaker_and_lexicon.py
│   ├── test_fusion.py       HMM, tracker, all three models, featurizer
│   ├── test_events.py       the full transfer lifecycle
│   └── test_e2e.py          whole pipeline on generated audio vs ground truth
├── data/synthetic/          generated corpus (wav + gold + script + telephony)
├── models/                  trained head + its meta.json
├── out/                     per-call timeline / events / segments / latency / summary
└── logs/                    structured per-hop JSONL traces
```

## Output format

Per call, in `out/`:

```
<call>.timeline.jsonl    one row per 0.5 s hop: state, posterior, confidence,
                         speaker, recent text, transfer_in_progress, latency
<call>.segments.jsonl    contiguous state runs — the at-a-glance view
<call>.events.jsonl      transfer_start / transfer_end / speaker_changed /
                         human_joined / ivr_exit, each with its evidence string
<call>.latency.jsonl     per-hop wall-clock, split by branch
<call>.summary.json      durations, speaker count, transfer counts, latency
                         percentiles, config fingerprint
```

And in `logs/<call>.log.jsonl`, a structured trace carrying every hop's named
features and the state model's per-feature contributions — so a call that went
wrong can be debugged without re-running it.

JSONL because these are append-only time series: a live deployment writes them
as the call happens, and a crashed process still leaves a valid, readable
partial file.

## Going live

`WavFileSource` is the only file-specific component. A live deployment
replaces it with a source that yields the same `Frame(t_s, remote, agent)`
objects from a websocket, and nothing downstream changes — every stage already
reads only the ring buffer, which cannot return a sample that has not arrived.
`tests/test_causality.py` proves that structurally: it writes a monotonically
increasing counter as the audio signal, so any returned sample exceeding the
write count would be, literally, from the future. A second test runs the same
audio prefix twice with different futures appended and requires identical
beliefs over the shared prefix.

Run `--realtime` to pace a recorded file at wall-clock speed and profile the
whole pipeline under genuine timing pressure.
