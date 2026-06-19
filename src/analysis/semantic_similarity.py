from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

'''Converts claims to sentence embeddings'''
def encode_claims(claims):
    return model.encode(claims)

'''Compute similarity between two embeddings'''
def compute_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]
