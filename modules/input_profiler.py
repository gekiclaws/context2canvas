import pandas as pd
import os
import logging
script_dir = os.path.dirname(__file__)  # directory of input_profiler.py

from modules.llm.openai_client import prompt_model

def main(filepath, supported_classes=["line-plot", "dot-plot", "vertical-bar-graph", "horizontal-bar-graph", "pie-chart"], context = ""):
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
    
    question_examples = [
        "What’s the average rating for products by brand?",
        "How often does each error code occur in the system logs?",
        "What are the proportions of different payment methods used by customers?",
        "What are the top 10 most common job titles?",
        "What percentage of total sales comes from each product category?",
        "How do different countries’ inflation rates change over the years?",
        "How does life expectancy vary across a set of countries?",
        "What’s the share of devices used to access the platform (mobile vs. desktop)?",
        "Which city has the highest average rental price?",
        "How does website traffic vary week by week?",
        "What’s the number of bugs reported per software module?",
        "How is the population distributed by age group?",
        "Which department has the highest employee turnover rate?",
        "How much revenue does each region generate?",
        "What’s the average salary per department?",
        "What’s the market share of different companies in the industry?",
        "How is revenue split among business units or departments?",
        "What portion of total sales came from each product category?",
        "Is there a relationship between hours studied and exam scores?",
        "How many students are in each major?",
        "How do sales trends compare between Product A and Product B?",
        "What are the trends in programming language popularity by year?",
        "Which product category has the most sales?"
    ]
    
    # Create an interesting data question based on the dataset characteristics
    q_message = f"""
    Only consider the graph types mentioned here: {supported_classes}.
    Create a single, interesting data question based on {columns} and {summary_stats}.
    Do not return anything besides the data question.
    Your answer should be a simple sentence. Format your response like {question_examples}
    """
    
    if context != "":
        q_message = f"The user provided this additional context, which should override anything else: {context}." + q_message

    question = prompt_model(q_message, 2.0)
    logging.basicConfig(filename='evaluation/question_results.log', level=logging.INFO)
    logging.info(f"Result: {question}")

    # Determine the best visualization type given the question and dataset metadata
    viz_type = prompt_model(
        f"What is the best visualization class we should use to characterize this problem, "
        f"given {question}, {columns}, and {summary_stats}? "
        f"Do not return anything besides the visualization type. "
        f"Only return a type listed in {supported_classes}", 2.0
    )
    
    return question, viz_type, columns, summary_stats, df

if __name__ == "__main__":
    data_path = os.path.join(script_dir, "data/pixar_films.csv")
    question, viz_type, columns, summary_stats, df = main(data_path)
    print("Data Question:", question)
    print("Visualization Type:", viz_type)
    print("Columns:", columns)
    print("Summary Stats:", summary_stats)