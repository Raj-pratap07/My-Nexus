"""
Offline, explainable grader for answers about a document's "Known limitations"
section. Pure functions: no I/O, no network, no model calls.

WHAT IT IS: a lexical proxy. It aligns each claim in an answer with the items
of the section by shared content terms, and reports every claim it cannot
align or finds suspicious, with the terms and source lines that drove each
decision, so a person can check the verdict. It does NOT understand meaning:
a faithful but heavily reworded claim can be reported as unsupported, and a
wrong claim that reuses the source's words can be accepted. Never treat a pass
as semantic proof; read ``needs_review`` and the alignments.

Pipeline
  1. ``parse_section``  -> the section's items (top-level "- " bullets,
     continuation lines joined).
  2. ``segment_answer`` -> independent claims. Answer list items are split
     into sentences, semicolon clauses, explicit additive joins ("and also",
     "additionally", ...) and comma + conjunction clauses (", and" / ", but"),
     so an unsupported addition cannot hide behind a supported sentence in
     the same bullet. A bare "and" is not split ("titles and metadata").
  3. ``grade_limitations`` aligns each claim with section items (>= 2 shared
     terms, or 1 shared distinctive term), then runs five checks:
       completeness     every item is covered by its aligned claims (>= 30%
                        of the item's terms, or >= 3 of them);
       unsupported      a substantive claim aligns with no item, or aligns
                        weakly (< 50% of its non-status terms come from the
                        item), or adds terms absent from the whole source
                        (a technical-looking token, a capitalised name not
                        at the claim's start, or >= 3 plain terms). Each is
                        classified outside-section (a source line outside
                        the section supports >= 60% of its terms; that line
                        is reported as evidence) or not-in-source;
       polarity_lost    a claim mentions a term the item negates ("No rate
                        limiting", "Not yet tested") with no negation or
                        restriction in the same clause, or asserts positive
                        status ("implemented", "supported", ...) for an item
                        that states a restriction, with no restriction itself;
       status_invented  an aligned claim gives a planned/future status the
                        item does not give;
       quote_fidelity   every quotation (12+ chars) is verbatim in the source.
     A "Status: ..." note, or a clause made only of status words ("it is
     planned"), inherits the alignment of the claim before it. The grader
     never assigns a status itself: it only compares the status words a claim
     uses with those in the matching item.

Matching rules (all deterministic, see ``_term_match``): words are lower-cased,
stop words and answer-discourse words removed, simple suffixes stripped; two
terms match when equal, when one is a prefix (>= 4 chars) of the other, or when
they share a 5-char prefix and both are >= 5 chars.

Known blind spots (see eval_fixtures/limitations_answers.json):
  * prohibitions worded without a negation ("keep the API off any network")
    read as a lost restriction;
  * a reversal that keeps a negation word elsewhere in the same clause, or a
    wrong claim built entirely from source vocabulary, can pass;
  * fewer than 3 new plain words in an aligned claim only reach needs_review;
  * pronoun-led claims are judged on their own words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from app.agent.quote_fidelity import extract_quotes, unmatched_quotes

DEFAULT_HEADING = "Known limitations"

# --- vocabulary --------------------------------------------------------------
_STOP = frozenset("""
a an the and or but if then so as of to in on at by for from with into onto
over under about against between through during before after above below up
down out off again further once here there when where why how all any both
each few more most other some such own same than too very s t can will just
should would could may might must shall do does did doing done be is are was
were been being have has had having it its it's itself they them their theirs
this that these those which who whom whose what i me my we our you your he
she his her him us also even yet per via etc e.g i.e eg ie one two three four
five six seven eight nine ten first second third use uses used using make whether
makes made get gets got
""".split())
# Words an answer uses to talk ABOUT the source rather than to make a claim.
_DISCOURSE = frozenset("""
readme known limitation limitations project nexus my-nexus system document
documents documented documentation section sections list listed lists item
items state states stated say says said mention mentions mentioned describe
describes described description according note notes noted following
currently current explicitly explicit status specified unspecified specify
summary means mean meaning point points source sources file files answer
explain explains explained regarding concerning
""".split())
# Lack / restriction markers (checked on raw lower-cased words).
_RESTRICTION = frozenset("""
not no never without lack lacks lacking cannot can't isn't aren't doesn't
don't hasn't haven't wasn't won't only single missing none absent limited
neither nor merely solely just nobody nothing nowhere
""".split())
_POSITIVE_STATUS_RE = re.compile(
    r"\b(implemented|supported|supports|available|complete|completed|fully|"
    r"functional|enabled|tested|working|works|provides|provided)\b", re.I)
_PLANNED_RE = re.compile(
    r"\b(planned|plans? to|future|roadmap|upcoming|coming soon|scheduled|"
    r"intended to|later (?:release|version)|"
    r"will be (?:added|implemented|supported|introduced|built|tested|fixed|available))\b",
    re.I)
_NOT_YET_RE = re.compile(r"\bnot\s+yet\b", re.I)
_NEGATION_WINDOW = 6  # words before a status word that can negate it
# True negations open a scope that runs to the next , ; : . in the item;
# "only", "single" etc. restrict but do not negate the terms around them.
_NEGATORS = frozenset("""
not no never without lack lacks lacking cannot can't isn't aren't doesn't
don't hasn't haven't wasn't won't neither nor nobody nothing nowhere none
""".split())
# Status vocabulary. Judged by the status checks, so it is left out of the
# share of a claim's terms that must come from its item. Stemmed with _stem
# (see _status_stems) so the comparison is exact, never prefix-based: "work"
# must not swallow "worker".
_STATUS_WORDS = """
plan plans planned planning future roadmap upcoming scheduled later release
released version implement implemented implementation support supported
supports available complete completed fully functional enabled works working
provides provided intended coming soon
""".split()
_STATUS_LABEL_RE = re.compile(r"^\s*(?:implementation\s+)?status\s*:", re.I)
_INFERENCE_RE = re.compile(r"^\s*(interpretation|inference|my reading)\s*:", re.I)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*")
_TECH_RE = re.compile(r"^(?:[A-Z]{2,}[A-Za-z0-9]*|[A-Za-z]+[A-Z][A-Za-z0-9]*|.*\d.*)$")
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s", "ly")
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[]?[A-Z0-9*_`])")
# Clause boundaries: semicolons, explicit additive joins, and a comma followed
# by a coordinating conjunction (", and" / ", but" / ", while"). A bare "and"
# is NOT split, so "titles and metadata" stays one claim.
_CLAUSE_SPLIT_RE = re.compile(
    r";\s+|\s*,?\s+\b(?:and also|but also|as well as|additionally|in addition|"
    r"furthermore|moreover|plus)\b,?\s+|,\s+(?:and|but|while|whereas)\s+(?:that\s+)?",
    re.I)

# --- alignment thresholds (documented in the report and the module doc) ------
MIN_SHARED_TERMS = 2          # or one DISTINCTIVE term, see _align
DISTINCTIVE_MIN_LEN = 5
MIN_CLAIM_SUPPORT = 0.5       # share of an aligned claim's terms found in its item
COVERAGE_THRESHOLD = 0.3      # an item is covered at >= 30% of its terms ...
COVERAGE_MIN_TERMS = 3        # ... or >= 3 of them (long items, short bullets)
OUTSIDE_SUPPORT_THRESHOLD = 0.6
NOVEL_PLAIN_FAIL = 3          # novel non-technical terms that fail a claim


# --- small helpers -----------------------------------------------------------
def _strip_markup(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"[*_`]+", "", text)


def _stem(word: str) -> str:
    w = word.lower()
    if w.endswith("'s"):
        w = w[:-2]
    for suffix in _SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    return w


def _raw_words(text: str) -> list[str]:
    words = []
    for token in _WORD_RE.findall(_strip_markup(text)):
        words.extend(p for p in token.split("-") if p)
    return words


def _terms(text: str) -> list[str]:
    """Content terms (stemmed), in order, duplicates removed."""
    out: list[str] = []
    for word in _raw_words(text.replace("-", " ")):
        low = word.lower()
        if low in _STOP or low in _DISCOURSE or low in _RESTRICTION:
            continue
        stem = _stem(low)
        if len(stem) >= 2 and stem not in out:
            out.append(stem)
    return out


def _term_match(a: str, b: str) -> bool:
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 4 and long_.startswith(short):
        return True
    return len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5]


def _shared(terms: Sequence[str], other: Sequence[str]) -> list[str]:
    return [t for t in terms if any(_term_match(t, o) for o in other)]


def _has_restriction(text: str) -> bool:
    return any(w.lower() in _RESTRICTION for w in re.findall(r"[A-Za-z']+", text))


def _unnegated(regex: re.Pattern[str], text: str) -> list[str]:
    """Matches of ``regex`` not preceded, within a few words, by a negation."""
    hits = []
    for m in regex.finditer(text):
        before = re.findall(r"[A-Za-z']+", text[: m.start()])[-_NEGATION_WINDOW:]
        if not any(w.lower() in _NEGATORS for w in before):
            hits.append(m.group(0))
    return hits


def _negated_terms(item: str) -> list[str]:
    """Content terms inside a negation's scope (to the next , ; : .)."""
    out: list[str] = []
    for clause in re.split(r"[,;:.]", _strip_markup(item)):
        words = re.findall(r"[A-Za-z']+", clause.replace("-", " "))
        for k, w in enumerate(words):
            if w.lower() in _NEGATORS:
                out.extend(t for t in _terms(" ".join(words[k + 1:])) if t not in out)
    return out


def _status_stems() -> frozenset[str]:
    return frozenset(_stem(w) for w in _STATUS_WORDS)


def _support_terms(terms: Sequence[str]) -> list[str]:
    status = _status_stems()
    return [t for t in terms if t not in status]


def _novel_tokens(claim: str, source_terms: Sequence[str],
                  source_lower: str) -> tuple[list[str], list[str]]:
    """(technical-looking tokens, plain terms) of ``claim`` absent from the source."""
    tech = []
    for k, token in enumerate(_WORD_RE.findall(_strip_markup(claim))):
        low = token.lower()
        if low in _STOP or low in _DISCOURSE or low in source_lower or token in tech:
            continue
        # Technical shape (OAuth, MFA, TLS1.3), or a capitalised word that is
        # not the claim's first word: a named thing the source never mentions.
        if _TECH_RE.match(token) or (k > 0 and token[0].isupper()):
            tech.append(token)
    tech_stems = {_stem(t) for t in tech}
    plain = [t for t in _terms(claim)
             if t not in tech_stems and not any(_term_match(t, s) for s in source_terms)]
    return tech, plain


# --- 1. section parsing ------------------------------------------------------
def parse_section(markdown: str, heading: str = DEFAULT_HEADING) -> list[str] | None:
    """Items of the ``## <heading>`` section, or None if the heading is absent.

    The section ends at the next heading of level 1 or 2. Each top-level
    "- " / "* " bullet is one item; indented or unmarked continuation lines
    are joined to it with single spaces. Text is returned as written
    (Markdown emphasis kept).
    """
    lines = markdown.splitlines()
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*#*\s*$", re.I)
    start = next((i for i, l in enumerate(lines) if pattern.match(l.strip())), None)
    if start is None:
        return None
    items: list[str] = []
    for line in lines[start + 1:]:
        if re.match(r"^#{1,2}\s", line):
            break
        if not line.strip() or re.match(r"^#{3,6}\s", line):
            continue  # blank line or sub-heading: never part of an item
        m = re.match(r"^[-*]\s+(.*)$", line)
        if m:
            items.append(m.group(1).strip())
        elif items:
            items[-1] = f"{items[-1]} {line.strip()}"
    return items


def _source_lines_outside(markdown: str, heading: str) -> list[str]:
    """Sentence-sized units of the source outside the graded section."""
    lines = markdown.splitlines()
    pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*#*\s*$", re.I)
    out, skipping = [], False
    for line in lines:
        if pattern.match(line.strip()):
            skipping = True
            continue
        if skipping and re.match(r"^#{1,2}\s", line):
            skipping = False
        if not skipping and line.strip():
            out.append(line.strip())
    # Re-join wrapped prose, then split into sentences.
    text = "\n".join(out)
    units = []
    for para in re.split(r"\n(?=\s*(?:[-*|#]|\d+\.)\s)|\n\s*\n", text):
        joined = " ".join(para.split())
        units.extend(s for s in _SENTENCE_SPLIT_RE.split(joined) if s.strip())
    return units


# --- 2. answer segmentation --------------------------------------------------
def segment_answer(answer: str) -> list[str]:
    """Split an answer into independently assessable claims.

    Units are list items (with their continuation lines) or prose paragraphs;
    Markdown headings and table rules are dropped. Each unit is then split
    into sentences, semicolon clauses and explicit additive joins.
    """
    units: list[str] = []
    current: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped or re.match(r"^#{1,6}\s", stripped) or re.match(r"^[|:\- ]+$", stripped):
            if current:
                units.append(" ".join(current))
                current = []
            continue
        if _LIST_MARKER_RE.match(line):
            if current:
                units.append(" ".join(current))
            current = [_LIST_MARKER_RE.sub("", line).strip()]
        else:
            current.append(stripped)
    if current:
        units.append(" ".join(current))

    claims: list[str] = []
    for unit in units:
        for sentence in _SENTENCE_SPLIT_RE.split(unit):
            for clause in _CLAUSE_SPLIT_RE.split(sentence):
                clause = clause.strip(" -–—:")
                if clause:
                    claims.append(clause)
    return claims


def _is_list_answer_claim(answer: str, claim: str) -> bool:
    for line in answer.splitlines():
        if _LIST_MARKER_RE.match(line) and claim[:40] in _LIST_MARKER_RE.sub("", line):
            return True
    return False


# --- 3. grading --------------------------------------------------------------
@dataclass
class ClaimAssessment:
    claim: str
    terms: list[str]
    aligned_items: list[int]            # 0-based section item indices
    shared_terms: dict[int, list[str]]
    verdict: str                        # supported | unsupported | non-substantive | review
    classification: str | None = None   # outside-section | not-in-source (unsupported only)
    evidence: str | None = None         # best source line outside the section
    novel_technical: list[str] = field(default_factory=list)
    novel_plain: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class GradeReport:
    section_found: bool
    items: list[str]
    claims: list[ClaimAssessment]
    coverage: list[float]
    missing: list[int]
    unsupported: list[ClaimAssessment]
    polarity_lost: list[dict]
    status_invented: list[dict]
    quotes_checked: int
    quotes_unmatched: list[str]
    needs_review: list[str]

    @property
    def failed_checks(self) -> list[str]:
        failed = []
        if not self.section_found:
            failed.append("section_missing")
        if self.missing:
            failed.append("completeness")
        if self.unsupported:
            failed.append("unsupported")
        if self.polarity_lost:
            failed.append("polarity_lost")
        if self.status_invented:
            failed.append("status_invented")
        if self.quotes_unmatched:
            failed.append("quote_fidelity")
        return failed

    @property
    def passed(self) -> bool:
        return not self.failed_checks

    def explain(self) -> str:
        """Human-readable report: every decision and the terms behind it."""
        out = [f"RESULT: {'PASS' if self.passed else 'FAIL'} "
               f"(lexical proxy, not semantic proof) failed={self.failed_checks}"]
        for i, item in enumerate(self.items):
            mark = "MISSING" if i in self.missing else "covered"
            out.append(f"  item {i + 1} [{mark} {self.coverage[i]:.2f}]: {item}")
        for c in self.claims:
            line = f"  claim [{c.verdict}] {c.claim!r}"
            if c.aligned_items:
                line += " -> items " + ", ".join(
                    f"{i + 1} {c.shared_terms[i]}" for i in c.aligned_items)
            if c.classification:
                line += f" ({c.classification}"
                line += f"; evidence: {c.evidence!r})" if c.evidence else ")"
            if c.novel_technical or c.novel_plain:
                line += f" novel={c.novel_technical + c.novel_plain}"
            out.append(line)
        for p in self.polarity_lost:
            out.append(f"  polarity lost: item {p['item'] + 1}: {p['reason']}")
        for s in self.status_invented:
            out.append(f"  status invented: item {s['item'] + 1}: {s['words']} in {s['claim']!r}")
        for q in self.quotes_unmatched:
            out.append(f"  quotation not verbatim in source: {q!r}")
        for r in self.needs_review:
            out.append(f"  REVIEW: {r}")
        return "\n".join(out)


def _align(claim_terms: Sequence[str], item_terms: Sequence[Sequence[str]]
           ) -> tuple[list[int], dict[int, list[str]]]:
    """Items a claim aligns with: >= MIN_SHARED_TERMS shared terms, or one
    shared term that is distinctive (long, and in no other item)."""
    aligned, shared_by_item = [], {}
    for i, terms in enumerate(item_terms):
        shared = _shared(terms, claim_terms)
        if not shared:
            continue
        others = [t for j, ts in enumerate(item_terms) if j != i for t in ts]
        distinctive = [t for t in shared if len(t) >= DISTINCTIVE_MIN_LEN
                       and not any(_term_match(t, o) for o in others)]
        if len(shared) >= MIN_SHARED_TERMS or distinctive:
            aligned.append(i)
            shared_by_item[i] = shared
    return aligned, shared_by_item


def grade_limitations(answer: str, source_text: str, *,
                      heading: str = DEFAULT_HEADING,
                      extra_quote_sources: Sequence[str] = ()) -> GradeReport:
    """Grade ``answer`` against the ``heading`` section of ``source_text``.

    ``extra_quote_sources`` are texts that may legitimately be quoted besides
    the source (for example the user's own question).
    """
    items = parse_section(source_text, heading)
    quotes = extract_quotes(answer)
    unmatched = unmatched_quotes(quotes, [source_text, *extra_quote_sources])
    if items is None:
        return GradeReport(False, [], [], [], [], [], [], [], len(quotes), unmatched,
                           [f"source has no '## {heading}' section; nothing graded"])

    item_terms = [_terms(i) for i in items]
    item_restricted = [_has_restriction(i) for i in items]
    item_negated = [_negated_terms(i) for i in items]
    item_planned = [bool(_PLANNED_RE.search(i)) for i in items]
    item_not_yet = [bool(_NOT_YET_RE.search(i)) for i in items]
    source_terms = _terms(source_text)
    source_lower = _strip_markup(source_text).lower()
    outside = _source_lines_outside(source_text, heading)

    assessments: list[ClaimAssessment] = []
    needs_review: list[str] = []
    previous_aligned: list[int] = []
    for claim in segment_answer(answer):
        terms = _terms(claim)
        aligned, shared = _align(terms, item_terms)
        inherited = False
        status_only = bool(terms) and not _support_terms(terms)
        if ((_STATUS_LABEL_RE.match(claim) or status_only)
                and not aligned and previous_aligned):
            # "Status: ..." or a clause made only of status words ("it is
            # planned") describes the claim before it.
            aligned, inherited = list(previous_aligned), True
            shared = {i: [] for i in aligned}
        a = ClaimAssessment(claim, terms, aligned, shared, verdict="supported")
        if inherited:
            a.notes.append(f"status note; inherits items {[i + 1 for i in aligned]}")
        support_terms = _support_terms(terms)
        weak = (aligned and not inherited and support_terms and
                max(len(shared[i]) for i in aligned) / len(support_terms) < MIN_CLAIM_SUPPORT)
        if not terms and not inherited:
            a.verdict = "non-substantive"
        elif aligned and not weak:
            # Includes inherited status notes: they may not add new content either.
            a.novel_technical, a.novel_plain = _novel_tokens(claim, source_terms, source_lower)
            if a.novel_technical or len(a.novel_plain) >= NOVEL_PLAIN_FAIL:
                a.verdict, a.classification = "unsupported", "not-in-source"
                a.notes.append("aligned claim adds terms absent from the source")
            elif a.novel_plain:
                a.verdict = "review"
                needs_review.append(f"{claim!r}: terms not in the source {a.novel_plain} "
                                    "(paraphrase or addition?)")
        else:
            best, best_support = None, 0.0
            for line in outside:
                support = len(_shared(terms, _terms(line))) / len(terms)
                if support > best_support:
                    best, best_support = line, support
            substantive = (weak or _is_list_answer_claim(answer, claim)
                           or _has_restriction(claim) or len(terms) >= 3)
            if weak:
                a.notes.append(
                    f"weakly aligned: under {MIN_CLAIM_SUPPORT:.0%} of its terms are in "
                    f"items {[i + 1 for i in aligned]}")
                a.aligned_items, a.shared_terms = [], {}
            if _INFERENCE_RE.match(claim):
                a.verdict = "review"
                needs_review.append(f"{claim!r}: labelled inference, not a source claim")
            elif not substantive:
                a.verdict = "review"
                needs_review.append(f"{claim!r}: aligns with no section item")
            else:
                a.verdict = "unsupported"
                if best is not None and best_support >= OUTSIDE_SUPPORT_THRESHOLD:
                    a.classification, a.evidence = "outside-section", best
                else:
                    a.classification = "not-in-source"
        assessments.append(a)
        if a.aligned_items and a.verdict != "unsupported":
            previous_aligned = a.aligned_items

    # Completeness: union of terms of the claims aligned to each item.
    coverage, missing = [], []
    for i, terms in enumerate(item_terms):
        claim_terms = [t for a in assessments if i in a.aligned_items
                       and a.verdict != "unsupported" for t in a.terms]
        covered = _shared(terms, claim_terms)
        cov = len(covered) / len(terms) if terms else 1.0
        coverage.append(cov)
        if cov < COVERAGE_THRESHOLD and len(covered) < COVERAGE_MIN_TERMS:
            missing.append(i)

    polarity_lost, status_invented = [], []
    for i in range(len(items)):
        aligned = [a for a in assessments if i in a.aligned_items and a.verdict != "unsupported"]
        if not aligned:
            continue
        if item_restricted[i]:
            for a in aligned:
                negated = _shared(a.terms, item_negated[i])
                if negated and not _has_restriction(a.claim):
                    polarity_lost.append({"item": i, "reason": (
                        f"{a.claim!r} mentions {negated}, which the item negates, "
                        "without any negation or restriction")})
                positive = _unnegated(_POSITIVE_STATUS_RE, a.claim)
                if positive and not _has_restriction(a.claim):
                    polarity_lost.append({"item": i, "reason": f"{a.claim!r} asserts "
                                          f"{positive} for a documented limitation"})
                elif positive:
                    needs_review.append(f"{a.claim!r}: positive status {positive} next to "
                                        f"a restriction (item {i + 1})")
        for a in aligned:
            planned = _unnegated(_PLANNED_RE, a.claim)
            if planned and not item_planned[i]:
                status_invented.append({"item": i, "claim": a.claim, "words": planned})
            if _NOT_YET_RE.search(a.claim) and not item_not_yet[i]:
                needs_review.append(f"{a.claim!r}: 'not yet' implies future intent the "
                                    f"item does not state (item {i + 1})")

    return GradeReport(
        section_found=True, items=items, claims=assessments, coverage=coverage,
        missing=missing, unsupported=[a for a in assessments if a.verdict == "unsupported"],
        polarity_lost=polarity_lost, status_invented=status_invented,
        quotes_checked=len(quotes), quotes_unmatched=unmatched, needs_review=needs_review,
    )
