import re

def clean_text(text):
    """
    Clean text: lowercase, remove extra whitespace, remove special chars.
    Keeps alphanumeric, spaces, and punctuation needed for parsing.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+', '', text)
    return text

def preprocess_claims(df):
    """Apply cleaning to all claims in dataframe."""
    df['claim_clean'] = df['claim'].apply(clean_text)
    return df
