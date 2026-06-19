import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """Extract named entities from text."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def add_entities_to_df(df):
    """Add extracted entities to dataframe."""
    df['entities'] = df['claim_clean'].apply(extract_entities)
    return df
