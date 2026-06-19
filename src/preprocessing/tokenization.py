import spacy

nlp = spacy.load("en_core_web_sm")

def tokenize_claim(text):
    """Tokenize text using spaCy."""
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def add_tokens_to_df(df):
    """Add tokenized version of claims to dataframe."""
    df['tokens'] = df['claim_clean'].apply(tokenize_claim)
    return df
