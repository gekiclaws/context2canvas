import os
script_dir = os.path.dirname(__file__)  # directory of input_profiler.py

import pandas as pd
from llm.openai_client import prompt_model

def main(filepath, supported_classes=["line-plot", "dot-plot", "vertical-bar-graph", "horizontal-bar-graph", "pie-chart"]):
    """
    Process an input CSV file to extract column information, generate summary statistics,
    and then use an LLM to create a data question and suggest a visualization type.

    Args:
        filepath (str): The path to the CSV file.
        supported_classes (list, optional): List of allowed visualization types.
                                             For example: ["bar", "line", "scatter", "histogram"].

    Returns:
        tuple: Contains:
            - question (str): A data question generated from the dataset.
            - viz_type (str): The suggested visualization type.
            - columns (dict): A dictionary mapping column names to their data types.
            - summary_stats (dict): Summary statistics of the dataset in dictionary form.
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Extract column names and their data types
    columns = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Generate summary statistics and convert to a dict for better readability in prompts
    summary_stats = df.describe().to_dict()
    
    # Create an interesting data question based on the dataset characteristics
    question = prompt_model(
        f"Create an interesting data question based on {columns} and {summary_stats}. "
        f"Do not return anything besides the data question."
    )
    
    # Determine the best visualization type given the question and dataset metadata
    viz_type = prompt_model(
        f"What is the best visualization class we should use to characterize this problem, "
        f"given {question}, {columns}, and {summary_stats}? "
        f"Do not return anything besides the visualization type. "
        f"Only return a type listed in {supported_classes}"
    )
    
    return question, viz_type, columns, summary_stats

if __name__ == "__main__":
    data_path = os.path.join(script_dir, "data/pixar_films.csv")
    question, viz_type, columns, summary_stats = main(data_path)
    print("Data Question:", question)
    print("Visualization Type:", viz_type)
    print("Columns:", columns)
    print("Summary Stats:", summary_stats)