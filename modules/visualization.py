import matplotlib.pyplot as plt
import pandas as pd

def render_visualization(generated_code, df=None, return_raw=False):
    """
    Executes the validated Python code for visualization, renders the resulting chart,
    and optionally returns the raw Python code.

    This function executes the provided Python code (assumed to produce a chart, for example
    via matplotlib). It then attempts to capture the active matplotlib figure as the resulting chart.
    Finally, it optionally returns the raw Python code for manual tweaking or transparency.

    Args:
        generated_code (str): The validated Python code to execute.
        return_raw (bool, optional): Whether to also return the raw code alongside the chart.
                                     Defaults to False.

    Returns:
        If return_raw is False:
            chart (matplotlib.figure.Figure or None): The rendered visualization.
        If return_raw is True:
            tuple: (chart, generated_code) where chart is as above.
    """
    # Create an isolated global namespace for exec
    global_namespace = {'df': df}

    # Execute the generated code safely (beware of executing untrusted code)
    try:
        exec(generated_code, global_namespace)
    except Exception as e:
        print("An error occurred while executing the generated code:", e)
        return (None, generated_code) if return_raw else None

    # Attempt to capture the current matplotlib figure,
    # assuming that the generated code produced a chart.
    try:
        chart = plt.gcf()
        # Render the chart. In interactive environments, this may display the chart immediately.
        plt.show()
    except Exception as e:
        print("Failed to render the visualization:", e)
        chart = None

    if return_raw:
        return chart, generated_code
    
    return chart

if __name__ == "__main__":
    # Test example: a simple generated code snippet that creates a plot.
    test_generated_code = """
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/data.csv")

def parse_price(x):
    if pd.isna(x):
        return None
    return float(x.replace('$', '').replace(',', ''))
for col in ['Our Price', 'Comp 1 Price', 'Comp 2 Price', 'Comp 3 Price', 'Comp 4 Price']:
    df[col] = df[col].map(parse_price)
plt.figure(figsize=(10, 6))

plt.scatter(df['Our Rank'], df['Our Price'], alpha=0.7, c='tab:blue', label='Our Price', edgecolors='w', s=80)
comp_colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple']
comp_cols = ['Comp 1 Price', 'Comp 2 Price', 'Comp 3 Price', 'Comp 4 Price']
for col, color in zip(comp_cols, comp_colors):
    subset = df[df[col].notna()]
    plt.scatter(subset['Our Rank'], subset[col], alpha=0.5, c=color, label=col, edgecolors='w', s=50)
plt.xlabel('Our Rank')
plt.ylabel('Price (USD)')
plt.title('How does Our Rank relate to Price compared to competitors?')
plt.legend(title='Price Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(sorted(df['Our Rank'].unique()))
plt.tight_layout()
plt.show()
"""
    result = render_visualization(test_generated_code, return_raw=True)
    if result:
        chart, raw_code = result
        print("Visualization rendered. Raw code:")
        print(raw_code)