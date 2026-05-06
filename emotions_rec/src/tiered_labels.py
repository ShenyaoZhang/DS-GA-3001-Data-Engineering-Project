"""
Label maps for a future multi-stage / 6-class emotions setup (dair-ai/emotion).

Not used by `main_cluster_emotion_binary.py` (current default: one-vs-rest **joy**).

Dataset label ids:
  0 sadness, 1 joy, 2 love, 3 anger, 4 fear, 5 surprise

Reserved mapping:
  Tier 1 — binary sentiment (BERT): negative vs positive.
  Tier 2 — fine emotion within branches (see ORIG_TO_* / POS_SUB_*).
"""

# Tier 1: original dataset id → 0 negative, 1 positive
EMOTION_TO_BINARY = {
    0: 0,  # sadness
    1: 1,  # joy
    2: 1,  # love
    3: 0,  # anger
    4: 0,  # fear
    5: 1,  # surprise (positive at tier 1 only)
}

# Tier 2a — negative branch: dataset id → sub-model index (0=sadness, 1=anger, 2=fear)
ORIG_TO_NEG_SUB = {0: 0, 3: 1, 4: 2}
NEG_SUB_TO_ORIG = {0: 0, 1: 3, 2: 4}

# Tier 2b — positive branch: dataset id → sub-model index (0=joy, 1=love only)
ORIG_TO_POS_SUB = {1: 0, 2: 1}
POS_SUB_TO_ORIG = {0: 1, 1: 2}

# Human-readable names for all dataset ids (used in reports)
EMOTION_NAMES = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

# Leaf emotions this hierarchy predicts (dataset ids); surprise is not predicted at tier 2
LEAF_DATASET_LABELS = frozenset({0, 1, 2, 3, 4})
