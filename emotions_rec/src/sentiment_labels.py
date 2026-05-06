"""
Label maps for 3-class sentiment training.

Dataset label ids (dair-ai/emotion):
  0 sadness, 1 joy, 2 love, 3 anger, 4 fear, 5 surprise

Pipeline sentiment ids:
  0 negative, 1 neutral, 2 positive
"""

EMOTION_TO_SENTIMENT = {
    0: 0,  # sadness -> negative
    1: 2,  # joy -> positive
    2: 2,  # love -> positive
    3: 0,  # anger -> negative
    4: 0,  # fear -> negative
    5: 1,  # surprise -> neutral
}

SENTIMENT_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}
