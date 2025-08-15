# inst414-final-project-adam-zaidi

Project Overview:
    
    Business Problem:
    - Corporate litigation trends are difficult to predict without centralized access to court data and an appropriate predictive model. This project addresses that gap by creating a reproducible pipeline that retrieves case metadata and prepares it for downstream analysis (ex: classification, outcome prediction).
    
    Datasets Used:
    - CourtListener API (w./ CAP data): Provides case opinions and metadata (case name, court, citation, and decision date).
    - Processed Dataset: Cleaned and structured version of extracted data (stored in data/processed/).

    Techniques Employed:
    - ETL Pipeline: Extract -> Transform -> Load using Python and pandas.
    - Data Cleaning: Handles missing values, standardizes text fields.
	- Model: Logistic Regression and Random Forest on outcome data (temporary data at the moment).
	- Visualization: Bar chart of case outcomes using matplotlib.

    Expected Outputs:
	- Processed dataset of corporate cases (data/processed/processed_data.csv)
	- Analytical models trained on processed data (temp code at the momoent for future predictive work)
	- Visualization of outcome distribution (data/outputs/outcome_bar_chart.png)


Setup Instructions:

1. Clone Repository
2. Create Virtual Environment
3. Install Dependencies
4. API Key Setup
    4a. please do not use my key or my email lol (i will know + this repo is private)
    4b. you will need to create an account on CourtListener
    4c. use the email from your account and the API key and replace values in 
5. Run project using main.py
6. Outputs found in data/outputs

Code Package Structure:

inst414-final-project-adam-zaidi/
│
├── analysis/               # Model training and evaluation scripts
│   ├── __init__.py
│   ├── evaluate.py          # Model evaluation functions (classification/regression)
│   └── model.py             # Model training logic
│
├── data/                   # Data storage folders
│   ├── extracted/           # Raw API extraction files
│   ├── model-eval/          # Model evaluation metrics and plots
│   ├── outputs/             # Final outputs for reporting
│   ├── processed/           # Processed/cleaned datasets
│   └── reference-tables/    # Static reference data
│
├── etl/                    # Extract, Transform, Load pipeline
│   ├── __init__.py
│   ├── extract.py           # API extraction logic
│   ├── features.py          # Feature engineering
│   ├── load.py              # Load data into storage
│   └── transform.py         # Data cleaning and transformation
│
├── logs/                   # Log files
│   └── pipeline.log         # Log output for ETL pipeline runs
│
├── utils/                  # Helper utilities
│   ├── __init__.py
│   ├── evaluate.py          # Alternate evaluation helper
│   ├── logging_setup.py     # Logger configuration
│   └── validators.py        # Data validation checks
│
├── vis/                    # Visualization scripts
│   ├── __init__.py
│   └── visualizations.py    # Plot creation and saving
│
├── .env                     # Environment variables (API keys, etc.)
├── main.py                  # Main entry point to run the ETL + analysis pipeline
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies

## How to Run
2. **Configure environment variables**  
   - Create a `.env` file in the root directory with API keys or credentials.

3. **Run the pipeline**  
   ```
   python main.py
   ```

4. **Check outputs**  
   - Extracted data: `data/extracted/`
   - Processed data: `data/processed/`
   - Model evaluation: `data/model-eval/`
   - Visualizations: `vis/`
