import os
import random
import pandas as pd

from modules.code_generation import generate_code
from modules.visualization import render_visualization

def test_execution_pass_rate(num_tests=5):
    # Load CSV file (adjust the relative path as needed)
    data_path = os.path.join(os.path.dirname(__file__), "../modules/data/pixar_films.csv")
    df = pd.read_csv(data_path)

    # Stub for examples mimicking the RAG response structure
    examples = {
        "ids": [["001", "002", "003"]],
        "documents": [[
            "{'title': {'text': 'Sample Title 1'}}",
            "{'title': {'text': 'Sample Title 2'}}",
            "{'title': {'text': 'Sample Title 3'}}"
        ]],
        "embeddings": None,
        "uris": None,
        "data": None,
        "metadatas": [[None, None, None]],
        "distances": [[1.0, 1.1, 1.2]],
        "included": ["distances", "documents", "metadatas"]
    }

    # Define a wider range of test parameters.
    test_params = [
        {
            "viz_type": "bar",
            "question": "What is the distribution of film ratings?",
            "columns": {"rating": "float", "title": "object"},
            "summary_stats": {"rating": {"mean": 7.5, "std": 1.2}}
        },
        {
            "viz_type": "line",
            "question": "How have film ratings changed over time?",
            "columns": {"release_year": "int", "rating": "float"},
            "summary_stats": {"rating": {"min": 5.0, "max": 9.0}}
        },
        {
            "viz_type": "scatter",
            "question": "Is there a relationship between budget and film ratings?",
            "columns": {"budget": "float", "rating": "float"},
            "summary_stats": {"rating": {"mean": 7.5}, "budget": {"mean": 50}}
        },
        {
            "viz_type": "histogram",
            "question": "What is the distribution of film release years?",
            "columns": {"release_year": "int"},
            "summary_stats": {"release_year": {"mean": 1990}}
        },
        {
            "viz_type": "pie",
            "question": "What is the share of films by rating category?",
            "columns": {"rating_category": "object", "count": "int"},
            "summary_stats": {}
        },
        {
            "viz_type": "bar",
            "question": "What is the trend of film budgets over time?",
            "columns": {"release_year": "int", "budget": "float"},
            "summary_stats": {"budget": {"mean": 60}}
        },
        {
            "viz_type": "scatter",
            "question": "How do film durations relate to ratings?",
            "columns": {"duration": "float", "rating": "float"},
            "summary_stats": {"rating": {"mean": 7.5}, "duration": {"mean": 120}}
        }
    ]

    num_success = 0
    for i in range(num_tests):
        # Select a random test parameter set
        params = random.choice(test_params)
        print(f"Running test {i+1} with parameters:\n  Viz Type: {params['viz_type']}\n  Question: {params['question']}\n  Columns: {params['columns']}\n  Summary Stats: {params['summary_stats']}\n")
        
        # Generate the code based on the current parameters
        generated_code = generate_code(
            params["viz_type"],
            params["question"],
            params["columns"],
            params["summary_stats"],
            df,
            examples
        )

        result = render_visualization(generated_code, df)
        if result:
            print(f"Test {i+1} succeeded: Chart generated successfully.\n")
            num_success += 1
        else:
            print(f"Test {i+1} failed: No chart was generated.\n")
            

    print(f"Summary: {num_success} out of {num_tests} tests succeeded.")

if __name__ == "__main__":
    test_execution_pass_rate()