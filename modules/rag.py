import os
script_dir = os.path.dirname(__file__)

import json
from langchain_text_splitters import CharacterTextSplitter
import chromadb

default_filepath = os.path.join(script_dir, "data/annotations.json")

def index_data(json_filepath=default_filepath):
    """
    Load annotation data from a JSON file and extract annotations and types.

    Args:
        json_filepath (str, optional): Path to the JSON file with annotations.
                                       Defaults to 'data/annotations.json'.

    Returns:
        tuple: A tuple containing a list of annotation texts (str) and a list of their corresponding types.
    """
    with open(json_filepath, 'r') as f:
        data = json.load(f)
    
    # Instantiate a text splitter (if you want to split long annotations into chunks).
    # Currently, the text splitter is not applied; uncomment the split line if needed.
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    annotations = []
    types = []
    for record in data:
        annotation_text = str(record.get("general_figure_info", ""))
        # To split the text into chunks, uncomment the next two lines.
        # chunks = text_splitter.split_text(annotation_text)
        # annotations.extend(chunks)
        annotations.append(annotation_text)
        types.append(record.get("type", ""))
    
    return annotations, types

def chunk_data(annotations, collection_name="c2c", max_documents=1000):
    """
    Create a ChromaDB collection and index a subset of annotation documents.

    Args:
        annotations (list): List of annotation texts.
        collection_name (str, optional): Name for the ChromaDB collection. Defaults to "c2c".
        max_documents (int, optional): Maximum number of documents to index. Defaults to 1000.

    Returns:
        Collection: The created ChromaDB collection with the added documents.
    """
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name=collection_name)
    
    # Limit the documents to index if necessary
    selected_docs = annotations[:max_documents]
    collection.add(
        documents=selected_docs,
        ids=[str(i) for i in range(len(selected_docs))]
    )
    
    return collection

def query_data(question, collection, n_results=3):
    """
    Query the provided ChromaDB collection using a text question.

    Args:
        question (str): The input query string.
        collection (Collection): The ChromaDB collection to be queried.
        n_results (int, optional): Number of returned results. Defaults to 3.

    Returns:
        dict: The query results (examples) returned by the collection.
    """
    examples = collection.query(
        query_texts=[question],  # Chroma will convert the text to an embedding.
        n_results=n_results
    )
    return examples

if __name__ == "__main__":
    # Example usage of the RAG module
    annotations, types = index_data()
    collection = chunk_data(annotations)
    sample_question = "What information does this annotation provide?"
    results = query_data(sample_question, collection)
    print("Query Results:", results)