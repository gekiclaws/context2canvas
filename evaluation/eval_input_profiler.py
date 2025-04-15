from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity_matrix(model, messages):
    embeddings = model.encode(messages)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix
    
def compute_diversity_score(similarity_matrix):
    # remove diagonal (self-similarity)
    sim_no_diag = similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)].reshape(similarity_matrix.shape[0], -1)
    avg_similarity = np.mean(sim_no_diag)
    diversity_score = 1 - avg_similarity
    return diversity_score

def question_diversity(scorer=SentenceTransformer('all-MiniLM-L6-v2'), log_file='evaluation/question_results.log'):
    try:
        with open(log_file, 'r') as f:
            logs = f.readlines()
        matrix = compute_similarity_matrix(scorer, logs)
        score = compute_diversity_score(matrix)
        return score
    
    except FileNotFoundError:
        print("Log file not found.")
        return []

if __name__ == "__main__":
    question_diversity()