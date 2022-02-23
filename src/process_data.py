import os
import pandas as pd


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace
       are all typical text processing steps."""
    import re

    def strip(text):
        return text.strip()
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def text_cleaning(text):
        text = re.sub('[^A-Za-z0-9]+', ' ', str(text).lower()).strip()
        return text

    def lower(text):
        return text.lower()

    return strip(white_space_fix(text_cleaning(lower(s))))


def read_utterance(utt_file):
    return pd.read_csv(utt_file)


def read_call_transcript(utt_file):
    return pd.read_csv(utt_file)

