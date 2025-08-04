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
    data/
        extracted/
        processed/
        outputs/
        reference-tables/
    etl/
        extract.py
        transform.py
        load.py
    analysis/
        model.py
    vis/
        visualizations.py
    main.py
    requirements.txt
    README.md