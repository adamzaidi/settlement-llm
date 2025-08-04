from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data
from analysis.model import run_models
from vis.visualizations import generate_visualizations

def main():
    extract_data()
    transform_data()
    df = load_data()

    # Model (a lot of temp code at the moment)
    run_models(df)
    
    generate_visualizations(df)

if __name__ == "__main__":
    main()