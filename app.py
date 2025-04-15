import os
import pandas as pd

# Import functions from your project modules.
from modules.input_profiler import main as run_input_profiler
from modules.rag import index_data, get_or_create_collection, query_data
from modules.code_generation import generate_code as run_code_generator
from modules.visualization import render_visualization
from evaluation.eval_input_profiler import main as run_metrics

# Import the generic metrics functions
from evaluation.metrics import set_metric, get_metric, load_metrics, increment_metric_by_1, compute_execution_pass_rate

def run_pipeline(dataset_path="modules/data/pokemon_df.csv"):
    """
    Executes the full data-to-visualization pipeline:
      1. Input Profiling – reads CSV, extracts types and summary stats,
         and uses an LLM to generate a question and determine a visualization type.
      2. RAG – indexes annotations and retrieves a collection from persistent storage.
      3. Code Generation – queries the collection to fetch examples and generate code.
      4. Visualization Execution – executes the generated code to render a chart.
      5. Metrics Computation – updates persistent metrics including execution pass rate.
    """
    
    # Step 1: Input Profiling
    supported_vis_types = ["bar", "line", "scatter", "histogram"]
    question, viz_type, columns, summary_stats, df = run_input_profiler(dataset_path, supported_vis_types)
    print("=== Input Profiling Completed ===")
    print("Data Question:", question)
    print("Visualization Type:", viz_type)
    
    # Step 2: RAG – Load or Create Persistent Collection
    annotations, _ = index_data()
    collection = get_or_create_collection(annotations)
    print("=== RAG Module: Data Indexed or Loaded from Cache ===")
    examples = query_data(question, collection)
    print("Examples Retrieved:", examples)
    
    # Step 3: Code Generation
    generated_code = run_code_generator(viz_type, question, columns, summary_stats, df, examples)
    print("=== Generated Code ===")
    print(generated_code)
    
    # Step 4: Visualization Execution
    print("=== Executing Generated Visualization Code ===")
    chart = render_visualization(generated_code, df=df)
    
    if chart:
        print("Chart rendered successfully!")
        success = True
    else:
        print("No chart was rendered.")
        success = False

    # Step 5: Update and Report Metrics
    print("=== Updating and Generating Metrics ===")
    compute_execution_pass_rate(success)
    updated_metrics = load_metrics()
    print("Updated metrics:", updated_metrics)
    
    # Optionally run additional metrics calculations.
   #  extra_metrics = run_metrics()
   #  print("Additional metrics:", extra_metrics)
    
if __name__ == "__main__":
    run_pipeline()