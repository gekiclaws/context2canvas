import argparse
import pandas as pd

# Import functions from your project modules.
from modules.input_profiler import main as run_input_profiler
from modules.rag import index_data, get_or_create_collection, query_data
from modules.code_generation import generate_code as run_code_generator
from modules.visualization import render_visualization

# Import the generic metrics functions
from evaluation.metrics import get_metric, compute_execution_pass_rate, compute_question_diversity_score, compute_retrieval_alignment_score

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

   # Step 0: Prepare parser for user input
   parser = argparse.ArgumentParser()
   parser.add_argument("-c",
                       "--context",
                  help="Additional information from the user to direct the model's output",
                  type=str)
   parser.add_argument("-m",
                  "--metrics",
                  action="store_true",
                  help="Show performance metrics")
   parser.add_argument("-sr",
                  "--skip rag",
                  action="store_true",
                  help="Show performance metrics")
   args = parser.parse_args()

   # maps arguments to their input values
   if args.context:
      context = args.context
   else:
      context = ""

   metrics_on = args.metrics

   # Step 1: Input Profiling
   print("=== Generating Input Profile ===")
   supported_vis_types = ["bar", "line", "scatter", "histogram"]
   question, viz_type, columns, summary_stats, df = run_input_profiler(dataset_path, supported_vis_types, context, metrics_on)
   print("=== Input Profiling Completed ===")
   print("Data Question:", question)
   print("Visualization Type:", viz_type)

   # Step 2: RAG – Load or Create Persistent Collection
   print("=== Indexing Vector Database ===")
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
   print("=== Updating Records ===")
   compute_metrics(success)
   if metrics_on:
      print("=== Generating Metrics ===")
      print("Trials:", round(get_metric("num_trials"), 4))
      print("Question diversity score:", round(get_metric("question_diversity_score"), 4))
      #  print("Retrieval alignment score:", round(get_metric("retrieval_alignment_score"), 4))
      print("Execution pass rate:", round(get_metric("execution_pass_rate"), 4))

def compute_metrics(code_executed):
   compute_question_diversity_score()
   compute_retrieval_alignment_score()
   compute_execution_pass_rate(code_executed)
    
if __name__ == "__main__":
   run_pipeline()