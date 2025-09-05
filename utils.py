import re

def preprocess_text(text):
    """
    Basic preprocessing: remove links, mentions, hashtags, special chars.
    """
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"#\w+", "", text)     # remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)  # keep only letters
    return text.strip().lower()
