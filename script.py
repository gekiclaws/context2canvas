from modules.input_profiler import main as run_input_profiler
from modules.rag import index_data, get_or_create_collection
from modules.code_generation import generate_code as run_code_generator
from modules.visualization import render_visualization

def run_pipeline():
    """
    Executes the full data-to-visualization pipeline:

    1. Input Profiling:
       - Reads a CSV dataset.
       - Extracts column type information and summary statistics.
       - Uses an LLM to generate a data question and determine a visualization type.

    2. RAG (Retrieval-Augmented Generation):
       - Loads annotation data from a JSON file.
       - Retrieves a persistent collection (indexed once on the first run).

    3. Code Generation:
       - Retrieves relevant examples by querying the collection.
       - Uses an LLM to generate Python visualization code based on the visualization type,
         data question, dataset columns, and summary statistics.

    4. Visualization Execution:
       - Executes the generated Python code to render and return the resulting chart.
       - Optionally returns the raw Python code for transparency or manual tweaking.

    The function prints progress messages at each step and displays the generated chart.
    """
    
    # Step 1: Input Profiling
    csv_filepath = "modules/data/pixar_films.csv"
    supported_vis_types = ["bar", "line", "scatter", "histogram"]
    question, viz_type, columns, summary_stats = run_input_profiler(csv_filepath, supported_vis_types)
    print("=== Input Profiling Completed ===")
    print("Data Question:", question)
    print("Visualization Type:", viz_type)
    
    # Step 2: RAG – Load or Create Persistent Collection
    annotations, _ = index_data()
    collection = get_or_create_collection(annotations)
    print("=== RAG Module: Data Indexed or Loaded from Cache ===")
    
    # Step 3: Code Generation
    generated_code = run_code_generator(viz_type, question, columns, summary_stats, collection)
    print("=== Generated Code ===")
    print(generated_code)
    
    # Step 4: Visualization Execution
    print("=== Executing Generated Visualization Code ===")
    # Change return_raw to True if you want to see the raw Python code as well.
    chart, raw_code = render_visualization(generated_code, return_raw=True)
    if chart:
        print("Chart rendered successfully!")
    else:
        print("No chart was rendered.")
    
if __name__ == "__main__":
    run_pipeline()