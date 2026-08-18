"""
Language-side evidence: IVR prompts, hold scripts, transfer announcements,
and markers of spontaneous human speech.

Language is the single strongest IVR signal available. Acoustically a good
TTS prompt and a calm representative can look very similar; lexically they are
nothing alike — "for pharmacy benefits, press or say one" is unambiguous where
the waveform is not. So this branch exists to feed the fusion layer strong
semantic priors, not to make the decision itself.

These are patterns, not a classifier, on purpose: they are inspectable, they
fire identically every run, and a reviewer can trace exactly which phrase
caused a label. Their weakness is paraphrase — they will miss "I'm going to
put you through to somebody in claims". `--text-encoder minilm` plus the
trained fusion head covers that case; the two are complementary, which is why
both the pattern scores *and* the text embedding go into the feature vector.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Pattern

from ..encoders.text_encoder import tokenize


def _compile(patterns: List[str]) -> List[Pattern]:
    return [re.compile(p) for p in patterns]


IVR_PATTERNS = _compile([
    r"\bpress\s+(or say\s+)?(one|two|three|four|five|six|seven|eight|nine|zero|\d)\b",
    r"\bfor\s+[\w\s]{2,25}?,?\s*(press|say)\b",
    r"\bplease (enter|say|state|provide)\b",
    r"\b(main|previous) menu\b",
    r"\bto repeat (this|these|the) (menu|options)\b",
    # "your call may be recorded" is a disclosure and belongs to IVR;
    # "your call is important to us" is a hold-queue script and is listed
    # under HOLD_PATTERNS instead. Keeping both here made every hold segment
    # with a spoken overlay read as IVR -- a measured error, not a guess.
    r"\byour call may be (recorded|monitored)\b",
    r"\bthank you for calling\b",
    r"\busing (the|your) (touch.?tone )?keypad\b",
    r"\bdid you say\b",
    r"\bi didn'?t (quite )?(get|catch) that\b",
    r"\b(member|group|provider|npi|tax) (id|identification) number\b",
    r"\bfollowed by the pound (key|sign)\b",
    r"\bin a few words,? (tell|say)\b",
    r"\bplease listen carefully as our (menu|options) have changed\b",

    # --- conversational / speech-recognition IVR -------------------------
    # The patterns above assume a keypad menu. The real payer call in this
    # repo has no keypad prompt anywhere in its first two minutes — it is a
    # speech IVR that asks questions in full sentences ("Am I speaking with a
    # health care provider?", "Does the member ID start with the letter R
    # followed by numbers?"). Written against that transcript, because the
    # keypad-menu assumption silently missed the entire IVR phase.
    r"\bcalls? may be (monitored|recorded)\b",
    r"\bif this is a (medical )?emergency\b",
    r"\bplease hang up and (call|dial)\b",
    r"\bam i speaking (with|to) (a|an|the)\b",
    r"\bthis line is for\b",
    r"\bdoes the (member|group|policy) id\b",
    r"\bwhat (is|are) the (first|last|next) \w+ (characters?|digits?|letters?|numbers?)\b",
    r"\bplease say the (letters|numbers|digits)\b",
    r"\bfor example,? [a-z]([- ][a-z]){1,4}\b",
    r"\bis that correct\b",
    r"\bare you calling (for|about)\b.*\bor\b",
    r"\byour call is being connected\b",
    # NOT a bare `(member|group|provider) (id|identification)` — a human rep
    # says "can I get the member ID" constantly, and the bare form fired IVR
    # on live representatives. The framed variants above ("does the member
    # id...", "...member identification number") carry the IVR sense without
    # catching ordinary conversation.
    r"\bgo(ing)? to (the |our )?website\b",

    # --- written against real transcripts from two payers ----------------
    # Everything below was added after reading the words the recogniser
    # actually produced on moments the gold calls IVR and we called human.
    # Twenty-six such moments in one 5-minute chunk alone: the words were
    # there and the patterns simply did not cover them. Each line here is a
    # phrase observed verbatim, not a guess about how an IVR might speak.

    # Recording / legal disclosure. We matched "may be monitored" but the
    # real prompts say "will be monitored", which missed the entire opening
    # disclosure of one call.
    r"\b(your )?calls? (will|may|might) be (monitored|recorded)\b",
    r"\bmonitored (and|or) recorded\b",
    r"\bby continuing with this call\b",
    r"\byou understand,? accept and agree\b",
    r"\bis not an offer of payment\b",
    r"\b(does not|doesn'?t) guarantee (coverage|payment)\b",
    r"\bsubject to all benefit plan terms\b",
    r"\bmember eligibility at the time\b",

    # Identifier requests without the word "number" -- "your NPI or tax ID"
    # is the whole prompt, and requiring "number" missed it.
    r"\byour (npi|tax id|provider id)\b",
    r"\bthe patient'?s \w+ id\b",
    r"\b(npi|tax id) or (npi|tax id)\b",

    # Speech-IVR processing and recovery lines.
    r"\b(just|one) a? ?moment,? (please)?\b",
    r"\bi found your record\b",
    r"\blet'?s get started\b",
    r"\bsorry,? i (couldn'?t|could not|didn'?t) find\b",
    r"\bwhat would you like\b",
    r"\bsay (claims|eligibility|benefits|pharmacy|authorization)\b",
    r"\btry another\b",
    r"\bdo you want to use the\b",

    # Menu routing that does not use the bare "press N" form.
    r"\bor press for\b",
    r"\bif your question is about\b",
    r"\bplease contact\b",
    r"\bpara espa\w*ol\b",
    r"\bmarque dos\b",

    # Connection / queue announcements spoken by the system.
    r"\bplease hold while your call\b",
    r"\bbeing connected to the\b",
    r"\bappropriate (blue cross|plan|department)\b",

    # Self-service deflection, a very common IVR block.
    r"\balternative way to get your questions? answered\b",
    r"\bchat with us\b",
    r"\bchat now under the\b",
    r"\bmember portal app\b",
    r"\bback of your (id )?card\b",
])

HOLD_PATTERNS = _compile([
    r"\b(please|kindly) (hold|stay on the line)\b",
    r"\byour call is important to us\b",
    r"\bstay on the line\b",
    r"\bthanks? for (holding|your patience|waiting)\b",
    r"\byour (call|estimated) wait time\b",
    r"\ball (of our )?(agents|representatives|specialists) are (currently )?(busy|assisting)\b",
    r"\byou are (number )?\w+ in (the |our )?queue\b",
    r"\bwe'?ll be with you (shortly|in a moment)\b",
    r"\bdo not hang up\b",
])

TRANSFER_PATTERNS = _compile([
    r"\b(let me|i'?ll|i am going to|i'?m going to|gonna) (get you (over )?to|transfer you|connect you|patch you|put you through)\b",
    r"\btransferr?ing you (now|over|to)\b",
    r"\bconnecting you (now|to|with)\b",
    r"\bplease hold (for|while (i|we))\b",
    r"\bone moment while (i|we) (transfer|connect|get)\b",
    r"\bi'?ll (get|bring) (a |my )?(supervisor|specialist|someone|somebody)\b",
    r"\b(stay|remain) on the line while (i|we)\b",
    r"\bthat'?s (handled by|a question for) (a )?(different|another)\b",
])

TRANSFER_FAIL_PATTERNS = _compile([
    r"\b(all|our) (agents|representatives) are busy\b.*\bi'?ll help\b",
    r"\bthey'?re (not available|unavailable|on another call)\b",
    r"\bi'?ll (go ahead and )?(help|assist|handle) (you|this|it) (myself|instead)\b",
    r"\bthe transfer (didn'?t|did not) (go through|work)\b",
    r"\bsorry (about that|for the wait),? (i'?m|let me) (help|assist)\b",
])

# Disfluencies, hedges and backchannels — hallmarks of unscripted speech.
# A recorded prompt essentially never contains them.
HUMAN_MARKERS = {
    "um", "umm", "uh", "uhh", "hmm", "mhm", "yeah", "yep", "okay", "ok", "gotcha",
    "sure", "right", "alright", "lemme", "sorry", "actually", "basically", "like",
    "well", "so", "let's", "i'm", "i'll", "we're", "you're", "that's", "gimme",
}
HUMAN_PHRASES = _compile([
    r"\b(let me|lemme) (check|see|pull|look)\b",
    r"\bone (second|sec|moment)\b",
    r"\bbear with me\b",
    r"\bcan you (repeat|say that again|spell)\b",
    r"\bmy (name|system) is\b",
    r"\bhow (are you|can i help)\b",
    r"\bwhat'?s the (patient|member|date)\b",
    r"\bi (see|show|have) (that|it|here)\b",
])


_PUNCT = re.compile(r"[^a-z0-9']+")


def normalize(text: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace — before any matching.

    A real recogniser inserts sentence punctuation wherever it hears a pause,
    and windowed decoding puts those pauses in arbitrary places. On the real
    call Whisper returned "Thank you. for calling Blue Card Eligibility",
    which the pattern `\\bthank you for calling\\b` does not match because of
    the period. Matching against normalized text costs nothing and removes a
    whole class of silent misses that only appear on real audio.
    """
    return _PUNCT.sub(" ", (text or "").lower()).strip()


def _score(text: str, patterns: List[Pattern]) -> float:
    """Saturating hit count — one match is strong evidence, three is not 3x."""
    hits = sum(1 for p in patterns if p.search(text))
    if hits == 0:
        return 0.0
    return float(min(1.0, 0.55 + 0.2 * hits))


@dataclass
class LexicalScores:
    ivr_prompt_prob: float
    hold_phrase_prob: float
    transfer_phrase_prob: float
    transfer_fail_prob: float
    human_spontaneous_prob: float
    hits: Dict[str, List[str]]


def score_text(text: str) -> LexicalScores:
    t = normalize(text)
    if not t.strip():
        return LexicalScores(0.0, 0.0, 0.0, 0.0, 0.0, {})

    toks = tokenize(t)
    marker_hits = [w for w in toks if w in HUMAN_MARKERS]
    marker_ratio = len(marker_hits) / max(len(toks), 1)
    human = min(1.0, 0.9 * min(marker_ratio / 0.12, 1.0) * 0.6 + _score(t, HUMAN_PHRASES) * 0.7)

    hits = {
        "ivr": [p.pattern for p in IVR_PATTERNS if p.search(t)],
        "hold": [p.pattern for p in HOLD_PATTERNS if p.search(t)],
        "transfer": [p.pattern for p in TRANSFER_PATTERNS if p.search(t)],
        "transfer_fail": [p.pattern for p in TRANSFER_FAIL_PATTERNS if p.search(t)],
        "human_markers": marker_hits[:6],
    }
    return LexicalScores(
        ivr_prompt_prob=_score(t, IVR_PATTERNS),
        hold_phrase_prob=_score(t, HOLD_PATTERNS),
        transfer_phrase_prob=_score(t, TRANSFER_PATTERNS),
        transfer_fail_prob=_score(t, TRANSFER_FAIL_PATTERNS),
        human_spontaneous_prob=float(min(human, 1.0)),
        hits={k: v for k, v in hits.items() if v},
    )
