import json

import os
script_dir = os.path.dirname(__file__)

from langchain_text_splitters import CharacterTextSplitter
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import chromadb

default_filepath = os.path.join(script_dir, "data/annotations.json")
index_directory = os.path.join(script_dir, "data/chroma_storage")

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

    
    # Instantiate a text splitter if needed for further splitting (currently not applied)
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    """

    annotations = []
    types = []
    for record in data:
        annotation_text = str(record.get("general_figure_info", ""))
        # You could process splitting here if needed:
        # chunks = text_splitter.split_text(annotation_text)
        # annotations.extend(chunks)
        annotations.append(annotation_text)
        types.append(record.get("type", ""))
    
    return annotations, types

def get_or_create_collection(annotations, collection_name="c2c", max_documents=1000, persist_directory=index_directory):
    """
    Retrieve a persistent ChromaDB collection using PersistentClient. If the collection
    does not already exist in the specified path, create it by indexing the supplied annotations,
    persist the new collection, and return it.

    Parameters:
        annotations (list): List of annotation texts.
        collection_name (str): Name of the collection. Defaults to "c2c".
        max_documents (int): Maximum number of documents to index. Defaults to 1000.
        persist_directory (str): Local path where the collection will be stored.
                                 If the path does not exist, it will be created.

    Returns:
        Collection: A ChromaDB collection object.
    """
    # Initialize the persistent client with the specified local storage directory.
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )

    # Check if the collection already exists.
    ## client.delete_collection(name=collection_name) ### delete vector DB for testing purposes
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' found.")
    except chromadb.errors.InvalidCollectionException:
        print(f"Collection '{collection_name}' not found. Creating a new collection.")
        collection = client.create_collection(name=collection_name)
        selected_docs = annotations[:max_documents]
        for ii in range(len(selected_docs)):
            collection.add(
                documents=selected_docs[ii],
                ids=str(ii)
            )
        print("New collection created and persisted.")

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
    # For testing: index data and get the persistent collection.
    annotations, types = index_data()
    collection = get_or_create_collection(annotations)
    sample_question = "What information does this annotation provide?"
    results = query_data(sample_question, collection)
    print("Query Results:", results)
