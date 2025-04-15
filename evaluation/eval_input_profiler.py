from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(model, messages):
    embeddings = model.encode(messages)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def question_diversity(scorer, log_file='module_results.log'):
    try:
        with open(log_file, 'r') as f:
            logs = f.readlines()
        return compute_similarity(scorer, logs)
    
    except FileNotFoundError:
        print("Log file not found.")
        return []

def main():
    scorer = BERTScorer(model_type='bert-base-uncased')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(question_diversity(model))