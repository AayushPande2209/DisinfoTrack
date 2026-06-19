import spacy

nlp = spacy.load("en_core_web_sm")

'''Extract subject, predicate, and object from a claim'''
def extract_triple(text):
    doc = nlp(text)
    
    subject = None
    predicate = None
    obj = None
    
    for token in doc:
        if token.dep_ == "nsubj":
            subject = token.text
        if token.dep_ == "ROOT":
            predicate = token.text
        if token.dep_ == "dobj":
            obj = token.text
            
    return (subject, predicate, obj)