import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_visualizations(df):
    """
    Generate bar chart showing count of corporate cases by court.
    Saves chart to data/outputs/outcome_bar_chart.png
    """
    os.makedirs("data/outputs", exist_ok=True)

    court_counts = df["court"].value_counts()
    court_counts.plot(kind="bar")
    plt.title("Corporate Cases by Court")
    plt.xlabel("Court")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("data/outputs/outcome_bar_chart.png")
    plt.close()

    print("Visualization saved to data/outputs/outcome_bar_chart.png")