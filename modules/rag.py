import json
import os
from pathlib import Path # Using pathlib for more robust path handling

# Ensure project root is correctly identified (assuming modules/rag.py is in 'modules')
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve() # This assumes 'modules' is directly under the project root

from langchain_text_splitters import CharacterTextSplitter
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import chromadb
from chromadb.utils import embedding_functions # Useful for explicitly defining embedding functions

# Define paths relative to the project root for consistency
default_filepath = project_root / "data" / "annotations.json"
index_directory = project_root / "data" / "chroma_storage"

def index_data(json_filepath=default_filepath):
    """
    Load annotation data from a JSON file and extract annotations and types.

    Args:
        json_filepath (Path, optional): Path to the JSON file with annotations.
                                        Defaults to 'data/annotations.json' relative to project root.

    Returns:
        tuple: A tuple containing a list of annotation texts (str) and a list of their corresponding types.
    """
    if not json_filepath.exists():
        print(f"Error: JSON file not found at {json_filepath}")
        return [], [] # Return empty lists to prevent further errors

    with open(json_filepath, 'r') as f:
        data = json.load(f)

    # Text splitter is commented out, but good to keep the context.
    # If you decide to use it, uncomment and integrate it.
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
        # If text splitting is enabled, `annotations.extend(chunks)` would be used here.
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
        persist_directory (Path): Local path where the collection will be stored.
                                  If the path does not exist, it will be created.

    Returns:
        Collection: A ChromaDB collection object.
    """
    # Create the persistence directory if it doesn't exist
    persist_directory.mkdir(parents=True, exist_ok=True)

    # Initialize the persistent client with the specified local storage directory.
    # ChromaDB will use default embedding function if not specified.
    # If you need a specific one (e.g., for Sentence Transformers), uncomment and configure:
    # default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(
        path=str(persist_directory), # Convert Path object to string for chromadb
        settings=Settings(), # Use default settings unless custom ones are needed
        # tenant=DEFAULT_TENANT, # These are usually not needed for PersistentClient unless multi-tenancy is specifically configured
        # database=DEFAULT_DATABASE,
    )

    try:
        # Try to get the collection
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' found.")
    except chromadb.errors.NotFoundError: # CORRECTED EXCEPTION TYPE
        print(f"Collection '{collection_name}' not found. Creating a new collection.")
        # When creating a collection, you can also specify the embedding function
        # collection = client.create_collection(name=collection_name, embedding_function=default_ef)
        collection = client.create_collection(name=collection_name)

        # Ensure documents are added correctly
        # ChromaDB's .add() expects lists for documents, metadatas, and ids
        selected_docs = annotations[:max_documents]
        
        # Prepare lists for batch addition
        documents_to_add = []
        ids_to_add = []
        # You can add metadata here if `types` or other info are relevant
        # metadatas_to_add = [] 

        for i, doc in enumerate(selected_docs):
            if doc.strip(): # Only add non-empty strings
                documents_to_add.append(doc)
                ids_to_add.append(f"doc_{i}") # Use a more descriptive ID than just 'i'

        if documents_to_add:
            collection.add(
                documents=documents_to_add,
                ids=ids_to_add
                # metadatas=metadatas_to_add # Uncomment and populate if needed
            )
            print(f"Added {len(documents_to_add)} documents to the new collection.")
        else:
            print("No valid documents to add to the new collection.")

    except Exception as e:
        # Catch any other unexpected errors during client or collection operations
        print(f"An unexpected error occurred with ChromaDB: {e}")
        raise # Re-raise the exception after logging

    return collection

def query_data(question, collection, n_results=2):
    """
    Query the provided ChromaDB collection using a text question.

    Args:
        question (str): The input query string.
        collection (Collection): The ChromaDB collection to be queried.
        n_results (int, optional): Number of returned results. Defaults to 2.

    Returns:
        dict: The query results (examples) returned by the collection.
    """
    if not question:
        print("Warning: Query question is empty.")
        return {}
        
    examples = collection.query(
        query_texts=[question],  # Chroma will convert the text to an embedding.
        n_results=n_results
    )
    return examples

if __name__ == "__main__":
    print("Starting ChromaDB RAG module...")
    # For testing: index data and get the persistent collection.
    annotations, types = index_data() # `types` is currently not used but returned.
    
    if annotations:
        print(f"Loaded {len(annotations)} annotations.")
        collection = get_or_create_collection(annotations)
        sample_question = "What information does this annotation provide?"
        
        # Check if the collection has documents before querying
        if collection.count() > 0:
            results = query_data(sample_question, collection)
            print("\n--- Query Results ---")
            if results and results.get('documents'):
                for i, doc in enumerate(results['documents'][0]): # Assuming one query text, so results[0]
                    print(f"Result {i+1}: {doc}")
            else:
                print("No relevant results found.")
        else:
            print("Collection is empty, cannot perform a query.")
    else:
        print("No annotations loaded. Cannot proceed with collection creation or querying.")

    print("\nChromaDB RAG module finished.")