import re

def load_conversation(file):
    """Reads a text file and returns its content as a string."""
    return file.getvalue().decode("utf-8")

def preprocess_text(text):
    """Cleans the text but retains punctuation relevant to conversation."""
    text = text.lower()
    text = re.sub(r"[^\w\s.!?]", "", text)  # Keep sentence-ending punctuation
    return text.strip()
