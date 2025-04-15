import os
import json

from evaluation.eval_input_profiler import question_diversity

# Define the path for the metrics file within the evaluation folder.
METRICS_FILE = os.path.join(os.path.dirname(__file__), "evaluation_metrics.json")

def load_metrics(filepath=METRICS_FILE):
  """
  Load metrics from the JSON file.
  Returns an empty dictionary if the file doesn't exist or if there's a JSON decoding error.
  """
  if os.path.exists(filepath):
      try:
          with open(filepath, "r") as f:
              return json.load(f)
      except (json.JSONDecodeError, OSError):
          return {}
  return {}

def save_metrics(metrics, filepath=METRICS_FILE):
  """
  Save the provided metrics dictionary to a JSON file.
  """
  try:
      with open(filepath, "w") as f:
          json.dump(metrics, f)
  except OSError as e:
      print(f"Error saving metrics: {e}")

def set_metric(metric, value, filepath=METRICS_FILE):
  """
  Set the given metric to the provided value.
  This behaves like a safe 'set_or_default' operation.
  
  Args:
    metric (str): The metric name (e.g., "num_trials", "num_success").
    value (int or float): The new value for the metric.
    filepath (str): Path to the metrics file (default uses METRICS_FILE).
  
  Returns:
    dict: The updated metrics dictionary.
  """
  metrics = load_metrics(filepath)
  metrics[metric] = value
  save_metrics(metrics, filepath)
  return metrics

def increment_metric_by_1(metric, filepath=METRICS_FILE):
  """
  Safely increments the given metric by 1. If the metric does not exist, it is initialized to 1.
  
  Args:
    metric (str): The metric to increment.
    filepath (str): Path to the metrics file.
  
  Returns:
    dict: The updated metrics dictionary.
  """
  metrics = load_metrics(filepath)
  # Use a safe default of 0 if the metric isn't set yet.
  current_value = metrics.get(metric, 0)
  metrics[metric] = current_value + 1
  save_metrics(metrics, filepath)
  return metrics

def get_metric(metric, default_value=0, filepath=METRICS_FILE):
  """
  Retrieve the value of a given metric, defaulting to default_value if not present.
  
  Args:
    metric (str): The metric name.
    default_value: Value to return if the metric is not found.
    filepath (str): Path to the metrics file.
  
  Returns:
    The current value of the metric or default_value if not found.
  """
  metrics = load_metrics(filepath)
  return metrics.get(metric, default_value)

def compute_execution_pass_rate(success, filepath=METRICS_FILE):
  """
  Computes the execution pass rate using 'num_success' and 'num_trials'
  and updates the 'execution_pass_rate' metric.
  
  Returns:
    dict: The updated metrics dictionary with the new 'execution_pass_rate'.
  """
  metrics = load_metrics(filepath)

  if success:
    increment_metric_by_1("num_success")
  increment_metric_by_1("num_trials")

  num_success = metrics.get("num_success", 0)
  num_trials = metrics.get("num_trials", 0)

  execution_pass_rate = (num_success / num_trials) if num_trials > 0 else 0
  return set_metric("execution_pass_rate", execution_pass_rate, filepath)

def compute_question_diversity_score(filepath=METRICS_FILE):
  return set_metric("question_diversity_score", question_diversity(), filepath)