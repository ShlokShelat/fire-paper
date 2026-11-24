"""FiRE — the 14-item ACE instrument used in the paper: the cluster -> item
table and the instrument's fixed parameters.

Items 1-9 are nine of the ten original ACE questions (Felitti et al., 1998);
items 10-14 follow the expanded 14-item version validated in India
(Trivedi et al., 2025). Supplementary Material, Section S2 lists the items
verbatim and the 69-entry cluster -> item table reproduced below.

This module is the single copy of the table. Worker 3 (ACE) projection
(fire_worker3_ace.py) and Worker 4 (ACE) scoring (fire_worker4_ace.py) both
import it, so the projection rule and the scoring rule cannot drift apart;
Worker 4's self-test also confirms that every entry resolves to its own item.

Resolution procedure (Supplementary Material, Section S2):
  1. Lower-case the cluster name and search it for each of the 69 entries
     as a substring, longest entry first, so a specific multi-word entry
     wins over a shorter one it contains.
  2. If no entry matches, one rule maps a name containing both a conflict
     word and a between-parents word to item 14 (Section S2 gives examples;
     the full word lists are _CONFLICT_WORDS and _BETWEEN_PARENTS_WORDS
     below).
  3. Otherwise the cluster is left unmapped (None) and reported, never
     guessed.
Constructs the 14-item instrument does not cover (household incarceration,
death of a parent or caregiver) are recognised first and resolve to None,
so they can never fall through to a substring match such as "parent".
"""

N_ITEMS = 14

# Childhood window, inclusive: age 18 is the last in-window year. Applied at
# ACE projection (fire_worker3_ace.py), not at the Stage 2-3 age check.
MIN_AGE, MAX_AGE = 0, 18

# The fixed 69-entry cluster -> item table (Supplementary Material, Section S2).
CLUSTER_ITEM = {
    "emotional abuse":               1,
    "verbal abuse":                  1,
    "verbal/emotional abuse":        1,
    "emotional/verbal abuse":        1,
    "psychological abuse":           1,
    "physical abuse":                2,
    "sexual abuse":                  3,
    "sexual molestation":            3,
    "inappropriate touch":           3,
    "emotional neglect":             4,
    "emotional unavailability":      4,
    "felt unloved":                  4,
    "feeling unwanted":              4,
    "feeling unloved":               4,
    "unwanted child":                4,
    "physical neglect":              5,
    "material neglect":              5,
    "parental separation":           6,
    "separation or divorce":         6,
    "parental divorce":              6,
    "divorce":                       6,
    "domestic violence against mother": 7,
    "domestic violence":             7,
    "abuse of mother":               7,
    "violence against mother":       7,
    "household substance":           8,
    "substance abuse":               8,
    "substance use":                 8,
    "alcoholism":                    8,
    "alcoholic":                     8,
    "parental alcoholism":           8,
    "problem drinking":              8,
    "drug use":                      8,
    "household mental illness":      9,
    "household mental":              9,
    "mental illness or suicide":     9,
    "mental illness":                9,
    "parental depression":           9,
    "household suicide":             9,
    "community violence":            10,
    "dangerous neighbourhood":       10,
    "dangerous neighborhood":        10,
    "neighbourhood violence":        10,
    "neighborhood violence":         10,
    "peer or sibling bullying":      11,
    "sibling bullying":              11,
    "peer bullying":                 11,
    "bullying":                      11,
    "peer rejection":                12,
    "peer rejection or isolation":   12,
    "rejection or isolation":        12,
    "social isolation":              12,
    "peer isolation":                12,
    "loneliness":                    12,
    "isolation":                     12,
    "childhood poverty":             13,
    "material deprivation":          13,
    "financial hardship":            13,
    "economic hardship":             13,
    "poverty":                       13,
    "parental conflict":             14,
    "interparental conflict":        14,
    "parental fighting":             14,
    "parental fights":               14,
    "parents fighting":              14,
    "parents fought":                14,
    "verbal conflict":               14,
    "marital conflict":              14,
    "spousal conflict":              14,
}

# Constructs outside the 14-item instrument. Recognised before the table so
# they resolve to None instead of falling through to an unrelated item
# (e.g. "death of parent" must not match on "parent").
NON_ACE_CLUSTERS = frozenset({
    "household incarceration", "incarceration", "imprisonment", "prison", "jail",
    "death of parent", "death of guardian", "death of parent or guardian",
    "death of caregiver", "death of mother", "death of father", "parental death",
    "loss of parent", "loss of mother", "loss of father",
})

_TABLE_LONGEST_FIRST = tuple(sorted(CLUSTER_ITEM, key=len, reverse=True))
_NON_ACE_LONGEST_FIRST = tuple(sorted(NON_ACE_CLUSTERS, key=len, reverse=True))

_DEATH_WORDS = ("death", "died", "passed away", "passing", "deceased")
_CAREGIVER_WORDS = ("parent", "mother", "father", "guardian", "caregiver", "mom", "dad")
_CONFLICT_WORDS = ("conflict", "fight", "fought", "arguing", "argument", "quarrel")
_BETWEEN_PARENTS_WORDS = ("parent", "marital", "spousal", "between")


def is_non_ace_cluster(cluster_name):
    """True if the cluster names a construct the 14-item instrument
    explicitly does not cover (incarceration, death of a parent/caregiver)."""
    cn = str(cluster_name or "").lower().strip()
    if any(token in cn for token in _NON_ACE_LONGEST_FIRST):
        return True
    return any(w in cn for w in _DEATH_WORDS) and any(w in cn for w in _CAREGIVER_WORDS) \
        and not any(token in cn for token in _TABLE_LONGEST_FIRST)


def cluster_to_item(cluster_name):
    """Resolve a cluster name to its ACE-14 item number, or None if the
    cluster is not part of the instrument."""
    cn = str(cluster_name or "").lower().strip()
    if not cn:
        return None
    for token in _NON_ACE_LONGEST_FIRST:
        if token in cn:
            return None
    for token in _TABLE_LONGEST_FIRST:
        if token in cn:
            return CLUSTER_ITEM[token]
    if any(w in cn for w in _DEATH_WORDS) and any(w in cn for w in _CAREGIVER_WORDS):
        return None
    if any(w in cn for w in _CONFLICT_WORDS) and any(w in cn for w in _BETWEEN_PARENTS_WORDS):
        return 14
    return None


def hint_items(entry):
    """ACE-14 item hints carried on an entry, if any.

    Only the explicitly namespaced field `ace14_items_hint` is read. The
    Worker 2 in this repository never writes it, so in a pipeline run every
    item comes from the cluster table; the field exists for state-space
    files that carry an explicit ACE-14 item on a state.

    The field `ace_items_hint` uses a different item numbering and is
    ignored, so it can never change an ACE-14 item.
    """
    out = []
    for h in (entry.get("ace14_items_hint") or []):
        try:
            n = int(h)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= N_ITEMS and n not in out:
            out.append(n)
    return out


def item_for_entry(entry, cluster_name):
    """The single ACE-14 item an entry projects to, and how it was resolved.

    Returns (item, source) where source is "cluster_table" or "hint", or
    (None, None). The cluster table is the projection rule; an ACE-14 hint
    is consulted only when the cluster name matches nothing in the table.
    """
    item = cluster_to_item(cluster_name)
    if item is not None:
        return item, "cluster_table"
    if is_non_ace_cluster(cluster_name):
        return None, None
    hints = hint_items(entry)
    if hints:
        return hints[0], "hint"
    return None, None
