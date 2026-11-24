#!/usr/bin/env python3
"""Non-LLM baselines for scoring the 14-item ACE questionnaire from flat notes.

  keyword   Published-lexicon rule baseline, the pre-LLM clinical NLP design
            of Bejan et al. (JAMIA 2018): ACE query terms + NegEx.
            * Terms (lexicon.csv): only terms that are (a) in Bejan et al.'s
              published ACE query (Table 2), (b) ACE concept strings released
              by Wu et al. (2022, github.com/knowlab/ACE-NLP), or (c) words of
              the ACE item text itself. Each term is assigned to the item whose
              wording it matches; plain inflections are added. Bejan terms
              that name no single item (child abuse, childhood trauma, child
              protective service, CPS ...) are not used.
            * Negation: NegEx (Chapman et al., J Biomed Inform 2001) with its
              published trigger list (negex_triggers.txt, unchanged, from
              github.com/chapmanbe/negex, negex.python). The PREN/POST scope
              logic is ported line for line from negex.py by P. Kang (2009).
              Terms with polarity "negated" come from items worded in the
              negative (e.g. item 4 "no one in your family loved you") and
              count only when NegEx marks them negated.
            * Person: items about a household member's acts or condition
              need a person word on the line (context column of lexicon.csv;
              lists below: ConText's family triggers, the person words of the
              item text, and the Indian kinship glossary given to every LLM
              baseline).
            * Age: only lines whose [Age ...] tag falls in 0-18.
            An item is yes if any line supports it; the line is its evidence.

  tfidf     TF-IDF (word 1-2 grams) + one logistic regression per item,
            class_weight="balanced", C=1.0, nothing tuned. Leave-one-
            individual-out: an individual's labels never train their model.
            Cites no evidence.

  majority  Each item set to its most common answer among the OTHER
            individuals (leave-one-out). The floor: a score with no reading.

Input: a folder of flat files, one per individual, lines like
    Section: text [Age 14]
ID = file name without --suffix (default "_flat.txt").
Reference: the JSON read by scripts/collect_answers.py,
    {ID: {"notes": "10010000100101", "notes+form": "...", "form": "..."}},
item 1 first. --ref-key picks the answers tfidf and majority learn from
("notes" for the notes-only rows, "notes+form" for the notes-plus-form rows).

Usage
    python3 nonllm_baselines.py keyword  NOTES_DIR OUT_DIR [--form --ref REF]
    python3 nonllm_baselines.py tfidf    NOTES_DIR OUT_DIR --ref REF
    python3 nonllm_baselines.py majority NOTES_DIR OUT_DIR --ref REF
    python3 nonllm_baselines.py formonly NOTES_DIR OUT_DIR --ref REF   (notes-plus-form floor)
    python3 nonllm_baselines.py score    OUT_DIR [OUT_DIR ...] --ref REF
    python3 nonllm_baselines.py evidence OUT_DIR evidence.csv
    python3 nonllm_baselines.py selftest       (or --selftest)

Paper concept: the non-LLM baselines of Supplementary Section S4 and Table
SVII. The outputs have the baselines' shape ({"items": {"1": {"present",
"source", "reasoning"}, ...}}), so scripts/collect_answers.py and
scripts/blind_review.py read them like any other system; `score` here is a
quick accuracy check only.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N_ITEMS = 14
MAX_AGE = 18

ITEM_TEXT = {
    1: "Parent or other adult in the household often swore at, insulted, put down, or humiliated you, or made you afraid of being physically hurt",
    2: "Parent or other adult in the household often pushed, grabbed, slapped, threw something at you, or hit you hard enough to leave marks or injure",
    3: "Adult or person 5+ years older touched or fondled you sexually, or tried to or had sex with you",
    4: "Often felt no one in the family loved you or thought you important or special, or the family was not close or supportive",
    5: "Not enough to eat, dirty clothes, no one to protect you, or parents too drunk or high to care for you",
    6: "Parents ever separated or divorced",
    7: "Mother or stepmother pushed, grabbed, slapped, had things thrown at her, kicked, bitten, hit, or threatened with a gun or knife",
    8: "Lived with a problem drinker, alcoholic, or someone who used street drugs",
    9: "Household member depressed, mentally ill, or attempted suicide",
    10: "Lived 2+ years in a dangerous neighborhood or saw people assaulted",
    11: "Other kids, including brothers or sisters, often hit, threatened, picked on, or insulted you",
    12: "Often felt lonely, rejected, or that nobody liked you (peer group)",
    13: "Family very poor or on public assistance for 2+ years",
    14: "Parents having physical or verbal fights all the time",
}

# Person-word lists for the context column of lexicon.csv.
_CONTEXT_FAMILY = ["mom", "dad", "sister", "brother", "aunt", "uncle", "grandfather",
                   "grandmother", "husband", "wife", "family history"]          # ConText family triggers
_ITEM_PERSONS = ["parent", "parents", "mother", "stepmother", "household", "family",
                 "brothers", "sisters"]                                        # ACE item text
_GLOSSARY = ["kaka", "mama", "chacha", "maasi", "bua", "nani", "dadi", "nana", "dada",
             "mil", "fil", "sil"]                                              # glossary given to LLM baselines
_FORMS = ["father", "stepfather", "papa", "mummy", "mumma"]                    # father; Hindi forms of mom/dad
CONTEXTS = {
    "household": _CONTEXT_FAMILY + _ITEM_PERSONS + _GLOSSARY + _FORMS,
    "parent": ["parent", "parents", "mother", "stepmother", "father", "stepfather",
               "mom", "dad", "papa", "mummy", "mumma"],
    "mother": ["mother", "stepmother", "mom", "mummy", "mumma"],
    "kids": ["kids", "kid", "brother", "brothers", "sister", "sisters"],   # item 11: other kids, brothers or sisters
    "neighborhood": ["neighborhood", "neighbourhood"],                     # item 10: neighborhood
    "none": [],
}


def _words_rx(words):
    return re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(w) + r"(?:'s)?" for w in words) + r")(?![a-z])", re.I)


CONTEXT_RX = {k: (_words_rx(v) if v else None) for k, v in CONTEXTS.items()}


def load_lexicon(path=os.path.join(HERE, "lexicon.csv")):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            item = int(r["item"])
            assert 1 <= item <= N_ITEMS and r["polarity"] in ("affirmed", "negated") and r["context"] in CONTEXTS, r
            rx = re.compile(r"(?<![a-z])(?:" + r["pattern"] + r")(?![a-z])", re.I)
            rows.append({"item": item, "rx": rx, "polarity": r["polarity"], "context": r["context"],
                         "pattern": r["pattern"], "source": r["source"]})
    return rows


# ---------------------------------------------------------------------------
# NegEx (definite negation, PREN + POST), ported from negex.py (P. Kang, 2009)
# ---------------------------------------------------------------------------
def load_negex(path=os.path.join(HERE, "negex_triggers.txt")):
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    lines.sort(key=len, reverse=True)                        # sortRules: longest first
    rules = []
    for line in lines:
        s = line.strip().split("\t")
        trig, tag = s[0], s[2]
        pattern = r"\b(" + r"\s+".join(trig.split()) + r")\b"
        rules.append((trig, tag, re.compile(pattern, re.I)))
    return rules


NEGEX = None


def negex_negated(sentence, phrase):
    """True if NegEx (negTagger with negP=False, as in the published
    wrapper.py) tags `phrase` in `sentence` as [NEGATED]. Exact port of
    negex.py: triggers are tagged first, then the phrase is located with the
    same regex and every copy of the matched string is marked."""
    global NEGEX
    if NEGEX is None:
        NEGEX = load_negex()
    filler = "_"
    s = sentence
    for trig, tag, rx in NEGEX:
        s = rx.sub(" " + tag + re.sub(r"\s+", filler, trig.strip()) + tag + " ", s)
    ph = re.sub(r"([.^$*+?{\\|()[\]])", r"\\\1", phrase)
    reP = re.compile(r"\b" + r"\W+".join(ph.split()) + r"\b", re.I)
    m = reP.search(s)
    if m:
        s = s.replace(m.group(0), "[PHRASE]" + re.sub(r"\s+", filler, m.group(0).strip()) + "[PHRASE]")
    toks = s.split()
    # [PREN] forward pass
    overlap = pren = 0
    sb = []
    for i in range(len(toks)):
        if toks[i][:6] == "[PREN]":
            pren, overlap = 1, 0
        if toks[i][:6] in ("[CONJ]", "[PSEU]", "[POST]", "[PREP]", "[POSP]"):
            overlap = 1
        if i + 1 < len(toks) and toks[i + 1][:6] == "[PREN]":
            overlap = 1
        if pren == 1 and overlap == 0:
            toks[i] = toks[i].replace("[PHRASE]", "[NEGATED]")
        sb.append(toks[i])
    # [POST] backward pass
    toks = list(reversed(sb))
    post = 0          # as in negex.py, overlapFlag is carried over, not reset
    out = []
    for i in range(len(toks)):
        if toks[i][:6] == "[POST]":
            post, overlap = 1, 0
        if toks[i][:6] in ("[CONJ]", "[PSEU]", "[PREN]", "[PREP]", "[POSP]"):
            overlap = 1
        if i + 1 < len(toks) and toks[i + 1][:6] == "[POST]":
            overlap = 1
        if post == 1 and overlap == 0:
            toks[i] = toks[i].replace("[PHRASE]", "[NEGATED]")
        out.insert(0, toks[i])
    return "[NEGATED]" in " ".join(out)


# ---------------------------------------------------------------------------
# Reading the notes
# ---------------------------------------------------------------------------
AGE_RX = re.compile(r"\[Age ([^\]]*)\]\s*$")
CHILDHOOD_RX = re.compile(r"\bchild(hood)?\b|school years|puberty|\bstd\b", re.I)


def parse_line(line):
    line = line.strip()
    m = AGE_RX.search(line)
    tag = m.group(1).strip() if m else ""
    body = line[:m.start()].strip() if m else line
    section, text = "", body
    if ": " in body:
        head, rest = body.split(": ", 1)
        if len(head) <= 120:
            section, text = head, rest
    return section, text, tag


def in_childhood(tag):
    """Tag in 0-18: the first age (or start of a span) is 18 or under.
    Unknown ages count only when the tag says childhood or school years."""
    t = tag.lower()
    nums = [int(n) for n in re.findall(r"\d+", re.sub(r"\d+(st|nd|rd|th)\s*std", "", t))]
    if t.startswith("unknown") or not nums:
        return bool(CHILDHOOD_RX.search(t))
    return nums[0] <= MAX_AGE


def sentences(text):
    return [s for s in re.split(r"[.;!?]", text) if s.strip()]


def read_notes(notes_dir, suffix):
    out = {}
    for path in sorted(glob.glob(os.path.join(notes_dir, "*" + suffix))):
        ident = os.path.basename(path)[: -len(suffix)]
        with open(path, encoding="utf-8") as f:
            out[ident] = [l for l in f.read().splitlines() if l.strip()]
    if not out:
        sys.exit(f"no files matching *{suffix} in {notes_dir}")
    return out


def read_ref(path, key):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if key not in v:
                sys.exit(f"error: reference entry {k!r} has no {key!r} answers "
                         f"(keys: {', '.join(sorted(v))})")
            v = v[key]
        out[k] = [int(c) for c in v]
    return out


# ---------------------------------------------------------------------------
# Keyword baseline
# ---------------------------------------------------------------------------
def line_hits(line, lexicon, age_filter=True):
    """{item: (term, source, sentence)} for the items this line supports.
    The sentence is the one that contained the matched term."""
    section, text, tag = parse_line(line)
    if age_filter and not in_childhood(tag):
        return {}
    full = f"{section} {text}"
    hits = {}
    for sent in sentences(text):
        for row in lexicon:
            if row["item"] in hits:
                continue
            ctx = CONTEXT_RX[row["context"]]
            if ctx is not None and not ctx.search(full):
                continue
            for term in dict.fromkeys(m.group(0) for m in row["rx"].finditer(sent)):
                neg = negex_negated(sent, term)
                if neg == (row["polarity"] == "negated"):
                    hits[row["item"]] = (term, row["source"], sent.strip())
                    break
    return hits


def run_keyword(notes, out_dir, forms=None, age_filter=True):
    lexicon = load_lexicon()
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for ident, lines in notes.items():
        items = {str(i): {"present": False, "source": "", "reasoning": "no matching term"} for i in range(1, N_ITEMS + 1)}
        for line in lines:
            for item, (term, src, sent) in line_hits(line, lexicon, age_filter).items():
                rec = items[str(item)]
                if not rec["present"]:
                    # Evidence = the matched sentence with its section header and age
                    # tag (the cited span), not the whole line; the line is kept for reference.
                    section, _, tag = parse_line(line)
                    span = (f"{section}: " if section else "") + sent + (f" [Age {tag}]" if tag else "")
                    rec.update(present=True, source=span, line=line.strip(), reasoning=f'matched "{term}" ({src})')
        if forms is not None:
            form = forms.get(ident)
            if form is None:
                print(f"warning: no form for {ident}, skipped", file=sys.stderr)
                continue
            # Notes-plus-form: the form's answer is kept unless the notes give
            # keyword evidence for the other answer (for a keyword matcher,
            # that can only be no -> yes). Kept answers cite the form.
            for i in range(1, N_ITEMS + 1):
                rec = items[str(i)]
                rec["form"] = form[i - 1]
                if form[i - 1] == 1:
                    note = rec["source"]
                    rec.update(present=True, changed=False, source="(form answer kept)",
                               reasoning="form says yes" + (f"; notes also matched: {note}" if note else ""))
                else:
                    rec["changed"] = bool(rec["present"])
        with open(os.path.join(out_dir, ident + ".json"), "w", encoding="utf-8") as f:
            json.dump({"items": items, "total_score": sum(r["present"] for r in items.values())}, f, indent=1, ensure_ascii=False)
        written += 1
    print(f"keyword: wrote {written} files to {out_dir} ({len(lexicon)} lexicon terms)")


# ---------------------------------------------------------------------------
# TF-IDF + logistic regression, leave-one-individual-out
# ---------------------------------------------------------------------------
def run_tfidf(notes, ref, out_dir):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        sys.exit("tfidf needs scikit-learn: pip install scikit-learn")
    ids = [i for i in notes if i in ref]
    missing = sorted(set(notes) - set(ref))
    if missing:
        print(f"warning: no reference for {missing}, skipped", file=sys.stderr)
    texts = {i: "\n".join(notes[i]) for i in ids}
    os.makedirs(out_dir, exist_ok=True)
    for test in ids:
        train = [i for i in ids if i != test]
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
        X = vec.fit_transform([texts[i] for i in train])
        x = vec.transform([texts[test]])
        items = {}
        for k in range(N_ITEMS):
            y = [ref[i][k] for i in train]
            if len(set(y)) == 1:
                pred, prob = y[0], float(y[0])
            else:
                clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
                clf.fit(X, y)
                prob = float(clf.predict_proba(x)[0][list(clf.classes_).index(1)])
                pred = int(prob >= 0.5)
            items[str(k + 1)] = {"present": bool(pred), "source": "", "reasoning": f"p(yes)={prob:.3f}; cites no evidence"}
        with open(os.path.join(out_dir, test + ".json"), "w", encoding="utf-8") as f:
            json.dump({"items": items, "total_score": sum(r["present"] for r in items.values())}, f, indent=1)
    print(f"tfidf: wrote {len(ids)} files to {out_dir} (leave-one-out over {len(ids)} individuals)")


def run_formonly(notes, forms, out_dir):
    """Notes-plus-form floor: the individual's own form answers, unchanged."""
    os.makedirs(out_dir, exist_ok=True)
    ids = [i for i in notes if i in forms]
    for test in ids:
        items = {str(k + 1): {"present": bool(forms[test][k]), "source": "(form answer)", "reasoning": "form unchanged"}
                 for k in range(N_ITEMS)}
        with open(os.path.join(out_dir, test + ".json"), "w", encoding="utf-8") as f:
            json.dump({"items": items, "total_score": sum(r["present"] for r in items.values())}, f, indent=1)
    print(f"formonly: wrote {len(ids)} files to {out_dir}")


def run_majority(notes, ref, out_dir):
    ids = [i for i in notes if i in ref]
    os.makedirs(out_dir, exist_ok=True)
    for test in ids:
        train = [i for i in ids if i != test]          # leave-one-out: no test label used
        yes = [sum(ref[i][k] for i in train) for k in range(N_ITEMS)]
        items = {str(k + 1): {"present": yes[k] * 2 > len(train), "source": "",
                              "reasoning": f"{yes[k]}/{len(train)} other individuals have this item"} for k in range(N_ITEMS)}
        with open(os.path.join(out_dir, test + ".json"), "w", encoding="utf-8") as f:
            json.dump({"items": items, "total_score": sum(r["present"] for r in items.values())}, f, indent=1)
    print(f"majority: wrote {len(ids)} files to {out_dir}")


# ---------------------------------------------------------------------------
# Scoring and evidence export
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def score(out_dirs, ref):
    for out in out_dirs:
        ok = n = fp = fn = 0
        per_item = [0] * N_ITEMS
        ids = []
        for ident, r in sorted(ref.items()):
            path = os.path.join(out, ident + ".json")
            if not os.path.exists(path):
                continue
            ids.append(ident)
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            got = [int(bool(d["items"][str(i)]["present"])) for i in range(1, N_ITEMS + 1)]
            for k, (a, b) in enumerate(zip(got, r)):
                n += 1
                ok += a == b
                fp += a > b
                fn += a < b
                per_item[k] += a == b
        if not n:
            print(f"=== {out}: no outputs matching the reference IDs")
            continue
        lo, hi = wilson(ok, n)
        print(f"=== {out}")
        print(f"  individuals={len(ids)}  decisions={n}  accuracy={100 * ok / n:.2f}% ({ok}/{n}; "
              f"Wilson 95% {lo:.1f}-{hi:.1f})  wrong yes={fp}  missed={fn}")
        print("  per item %: " + "  ".join(f"{k + 1}:{100 * c / len(ids):.0f}" for k, c in enumerate(per_item)))
        missing = sorted(set(ref) - set(ids))
        if missing:
            print(f"  note: {len(missing)} reference IDs have no output")


def export_evidence(out_dir, csv_path):
    rows = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["individual", "item", "item_text", "cited_evidence", "matched", "form", "changed", "grounded (yes/no)", "error_mode"])
        for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
            with open(path, encoding="utf-8") as g:
                d = json.load(g)
            for i in range(1, N_ITEMS + 1):
                rec = d["items"][str(i)]
                # Only answers that rest on note evidence are exported: every yes
                # in notes-only runs, and only changed answers in form runs.
                if rec["present"] and rec.get("changed", True):
                    w.writerow([os.path.basename(path)[:-5], i, ITEM_TEXT[i], rec.get("source", ""), rec.get("reasoning", ""),
                                rec.get("form", ""), rec.get("changed", ""), "", ""])
                    rows += 1
    print(f"evidence: {rows} rows written to {csv_path}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def selftest():
    lex = load_lexicon()
    cases = [
        ("Abuse: Father used to hit her with a belt [Age 8]", {2}),
        ("Abuse: Father used to hit her with a belt [Age 25]", set()),                  # outside 0-18
        ("Abuse: Teacher used to hit her [Age 8]", set()),                               # no household person
        ("History: Patient denies sexual abuse [Age 12]", set()),                         # NegEx PREN
        ("History: Sexual abuse by cousin [Age 12]", {3}),
        ("Family: No one in the family loved her [Age 10]", {4}),                         # negated-polarity term
        ("Family: Parents divorced [Age 9]", {6}),
        ("Family: Father drunk every night [Age unknown — childhood]", {8}),
        ("Family: Father drunk every night [Age unknown]", set()),                       # unknown age
        ("School: Bullied by classmates [Age 11]", {11}),
        ("School: Felt lonely in school [Age 6–12]", {12}),
        ("Home: Parents fought all the time [Age 14 onward]", {14}),
        ("Home: Mother was slapped by father [Age 7]", {2, 7}),                            # keyword limits: 2 also fires
    ]
    bad = 0
    for line, want in cases:
        got = set(line_hits(line, lex))
        flag = "ok " if got == want else "BAD"
        bad += got != want
        print(f"{flag} {sorted(got)!s:10} want {sorted(want)!s:10} {line}")
    assert in_childhood("12–16") and in_childhood("14 onward") and not in_childhood("19 — current")
    assert in_childhood("unknown — from 3rd std") and not in_childhood("unknown — college years")
    print("selftest:", "PASS" if not bad else f"{bad} FAIL")
    return bad == 0


def main():
    if "--selftest" in sys.argv[1:]:          # same entry point as the other scripts
        sys.exit(0 if selftest() else 1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["keyword", "tfidf", "majority", "formonly", "score", "evidence", "selftest"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--ref")
    ap.add_argument("--ref-key", default="notes", choices=["notes", "notes+form"],
                    help="tfidf, majority, score: which reference answers to use")
    ap.add_argument("--suffix", default="_flat.txt")
    ap.add_argument("--form", action="store_true", help="keyword: notes-plus-form condition (form from --ref)")
    ap.add_argument("--no-age", action="store_true", help="keyword: no 0-18 age filter (as the plain LLM baselines get no age instruction)")
    a = ap.parse_args()

    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    if a.cmd == "score":
        if not a.ref or not a.paths:
            sys.exit("usage: score OUT_DIR [...] --ref REF")
        return score(a.paths, read_ref(a.ref, a.ref_key))
    if a.cmd == "evidence":
        if len(a.paths) != 2:
            sys.exit("usage: evidence OUT_DIR evidence.csv")
        return export_evidence(*a.paths)
    if len(a.paths) != 2:
        sys.exit(f"usage: {a.cmd} NOTES_DIR OUT_DIR")
    notes = read_notes(a.paths[0], a.suffix)
    if a.cmd == "keyword":
        forms = None
        if a.form:
            if not a.ref:
                sys.exit("--form needs --ref")
            forms = read_ref(a.ref, "form")
        return run_keyword(notes, a.paths[1], forms, age_filter=not a.no_age)
    if not a.ref:
        sys.exit(f"{a.cmd} needs --ref")
    if a.cmd == "formonly":
        return run_formonly(notes, read_ref(a.ref, "form"), a.paths[1])
    ref = read_ref(a.ref, a.ref_key)
    (run_tfidf if a.cmd == "tfidf" else run_majority)(notes, ref, a.paths[1])


if __name__ == "__main__":
    main()
