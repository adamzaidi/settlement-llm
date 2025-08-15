# inst414-final-project-adam-zaidi
# CourtListener Final Project - Part 3
## Project Overview

This project implements an ETL pipeline, data processing, model evaluation, and visualizations for CourtListener data.


## Directory Structure

```
INST414-Final-Project/
│
├── .venv/                  # Virtual environment (dependencies installed here)
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
```

## How to Run
1. **Set up environment**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   pip install -r requirements.txt
   ```

<<<<<<< HEAD
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
=======
>>>>>>> 309d704 (part 3)
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
