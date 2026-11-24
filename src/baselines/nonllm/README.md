# Non-LLM baselines for ACE-14 scoring

The earlier keyword, negation and classifier systems for ACEs in clinical notes cannot be run here as released (see the table below). These baselines therefore reuse published components as closely as possible, and add nothing of our own beyond what is listed below.

## The baselines

**`keyword`: published ACE terms + NegEx.** This follows the design of Bejan et al. (JAMIA 2018, doi:10.1093/jamia/ocx059): ACE query terms plus the NegEx negation algorithm. It has four parts:
- **Terms (`lexicon.csv`, 94 rows).** Each term comes from exactly one of three sources, recorded in the `source` column:
  1. Bejan et al.'s published ACE query terms (their Table 2);
  2. the ACE concept strings released by Wu, Smith and Wu (2022), at github.com/knowlab/ACE-NLP (`Reddit-MH/aces_reddits_mental_health_322_ACEs.zip`);
  3. words of the ACE item text itself.

  Each term is assigned to the item whose wording it matches, and plain inflections are added (hit, hits, hitting). The Bejan terms that name no single item are not used: child abuse, childhood trauma, child protective service and the CPS terms. Neither are SemEHR fragments such as a bare "abuse", or the abbreviations "aa" and "aca".
- **Negation.** NegEx (Chapman et al., J Biomed Inform 2001) with its published trigger list (`negex_triggers.txt`, unchanged, from github.com/chapmanbe/negex), in the definite-negation setting (negP=False) of its `wrapper.py`. The tagging logic is an exact port of the published `negex.py`. On NegEx's 2,376 annotated test sentences it gives the same answer as an independent transcription of `negex.py` on all 2,376, and both agree with the human labels on 97.2%.
- **Person.** Items about a household member's acts or condition (items 1, 2, 5 drunk/high, 6, 7, 8, 9, 14) need a person word on the line. The person words come from:
  - ConText's family triggers;
  - the person words in the item text;
  - the Indian kinship glossary that every LLM baseline also receives.

  Our only additions are "father", "stepfather", and the Hindi forms papa, mummy and mumma.
- **Age.** Only lines whose `[Age …]` tag falls in 0–18 are used. Unknown ages are used only when the tag says childhood or school years. `--no-age` turns this off. The plain LLM baselines got no age instruction, so report both versions.

**`tfidf`: trained model.** TF-IDF on 1–2 word phrases, plus one logistic regression per item (balanced classes, default settings, nothing tuned). It is trained and tested leave-one-individual-out on the reference. It cites no evidence.

**`majority`: floor.** Each item gets its most common answer among the *other* individuals (leave-one-out), so no test label is used. When an item is exactly half yes, leave-one-out always predicts the opposite of the left-out person, which can score 0% on that item.

**`formonly`: notes-plus-form floor.** The individual's form, unchanged.

**`keyword --form`.** The notes-plus-form version of `keyword`: it keeps the form's answers and changes an item from no to yes where the notes give keyword evidence. Kept answers cite the form, and each item records `form` and `changed`. `evidence` exports only the changed answers, since those are the ones that can be unsupported changes (E6).

### Known limitations

- **Terms come only from the published sources.** Common words such as "beat" are therefore absent. NegEx's published triggers do not include "never", "nobody" or "n't".
- **Item 5 is worded with "and"** (no food, dirty clothes *and* no protection), but is matched as "or", as ACE scoring is usually applied.
- **The person check reads the whole line, including its section header.** This is lenient to the baseline.

### Published systems we checked and could not run

| System | Why it cannot be run |
|---|---|
| Annapragada et al. 2021 (PLoS ONE) | The rule model needs `AllRules.csv`, which was never released. The bag-of-words model needs a vocabulary fitted on their private notes. It covers physical abuse only. |
| Wu et al. 2022 | Needs a UMLS licence and SemEHR, which is archived, and it has no item mapping. Its released concept strings are used above. |
| Negriff et al. 2023 | Terms not released. |
| Bejan et al. 2018 | Ranks patients for "any ACE". Its terms and NegEx are used above. |

## Run

```
pip install scikit-learn            # only tfidf needs it
P=python3; B=src/baselines/nonllm/nonllm_baselines.py
N=NOTES_FOLDER; R=reference.json
$P $B --selftest
$P $B keyword  $N out/keyword
$P $B keyword  $N out/keyword_noage --no-age
$P $B keyword  $N out/keyword_form --form --ref $R
$P $B tfidf    $N out/tfidf    --ref $R --ref-key notes
$P $B majority $N out/majority --ref $R --ref-key notes
$P $B tfidf    $N out/tfidf_form    --ref $R --ref-key notes+form
$P $B majority $N out/majority_form --ref $R --ref-key notes+form
$P $B formonly $N out/formonly --ref $R
$P $B evidence out/keyword keyword_evidence.csv
```

- **Input files.** `NOTES_FOLDER` holds one flat file per individual (the preprocessor's `<id>_flat.txt`). Use `--suffix` if your names differ.
- **Reference.** The JSON used by `scripts/collect_answers.py`: `{"P01": {"notes": "1001…", "notes+form": "…", "form": "…"}}`, item 1 first. `--ref-key` picks the answers the trained and majority baselines learn from: `notes` for the notes-only rows of Table SVII, `notes+form` for the notes-plus-form rows; `--form` and `formonly` read `form`.
- **Scoring.** Collect the outputs with `scripts/collect_answers.py` and score them with `scripts/evaluate_judgements.py` (after judging) and `scripts/clinical_measures.py`, like every other system.
- **Grounding.** `evidence` writes every line the keyword baseline cited; the psychotherapists judge them blind with the other systems (`scripts/blind_review.py`).
- **Licence.** `negex_triggers.txt` is the NegEx trigger list from github.com/chapmanbe/negex, unchanged, released under the Apache License 2.0 (`LICENSE-NEGEX.txt`); the lexicon's sources are listed per row in `lexicon.csv`.
