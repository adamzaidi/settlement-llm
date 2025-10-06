CURRENTLY IN DEVELOPMENT - LLM to determine case outcome likelihood
**Author:** Adam Zaidi  
**Date:** August 2025  

---

## Project Overview
This project builds a full **ETL + ML pipeline** around U.S. court opinions using the **CourtListener API**. I extract opinion data about corporations, enrich it with court/citation metadata, transform it into a clean tabular dataset, and then train simple classification models to predict case outcomes.  

The project is designed to answer a **business problem**: *“Given court opinions about corporations, what can I learn about how cases resolve (Win/Loss/Settlement) and how outcomes vary by court?”*  

The pipeline outputs:
- Cleaned and enriched case data  
- Outcome label assignments (coarse + fine)  
- Evaluation artifacts for classification models  
- Visualizations to support non-technical stakeholders  

---

## How to Run the Pipeline

1. **Clone the repo & create a virtual environment**
   ```bash
   git clone <your_repo_url>
   cd <repo_name>
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set environment variables** (create a `.env` file or export manually):
   ```
   COURTLISTENER_API_KEY=YOUR_KEY_HERE
   COURTLISTENER_EMAIL=your@email.com
   ```

3. **Run end-to-end pipeline**
   ```bash
   python main.py
   ```

---

## Pipeline Steps

1. **Extract**  
   - Source: CourtListener API (`opinions/` endpoint)  
   - Output: `data/extracted/raw_data.csv`  

2. **Transform**  
   - Cleans types, fills missing values  
   - Enriches with court names, citations, and opinion text signals  
   - Assigns **outcome_code** (coarse) and **outcome_code_fine** (fine)  
   - Output: `data/processed/processed_data.csv`  

3. **Load**  
   - Reads processed CSV into a pandas DataFrame for analysis  

4. **Modeling**  
   - Trains Logistic Regression and Random Forest (or skips gracefully if not enough labels)  
   - Always writes an evaluation artifact to `data/model-eval/`  
   - Evaluation includes classification reports, confusion matrices, and summary JSON  

5. **Visualizations**  
   - Generates outcome distributions, top courts, Pareto charts, trends over time, and word count plots  
   - Output: `data/outputs/*.png`  

6. **Validation**  
   - Runs lightweight schema + quality checks after load  
   - Logs missing columns, invalid codes, or duplicate IDs  

---

## Outputs

- `data/extracted/raw_data.csv` – raw API dump  
- `data/processed/processed_data.csv` – enriched dataset  
- `data/model-eval/` – evaluation reports, confusion matrices, summary JSON  
- `data/outputs/` – visualization PNGs  
- `logs/pipeline.log` – log file of pipeline run  

---

## Repo Structure

```
.
├── analysis/
│   └── model.py              # Train & evaluate classifiers
├── etl/
│   ├── extract.py            # Extract raw data from CourtListener
│   ├── transform.py          # Clean + enrich data
│   ├── load.py               # Load processed data
│   └── enrich_helpers.py     # Court/citation inference utilities
├── utils/
│   ├── logging_setup.py      # Logging configuration
│   └── validators.py         # Post-load validation checks
├── vis/
│   └── visualizations.py     # Generates charts
├── evaluate.py               # Extra model evaluation helpers
├── main.py                   # Pipeline runner
├── requirements.txt          # Dependencies
├── data/
│   ├── reference-tables/     # Data dictionaries (kept in repo)
│   └── (other subdirs ignored in git)
├── logs/                     # Runtime logs (gitignored)
└── README.md
```

---

## Data Dictionaries

See `data/reference-tables/`:
- `raw_data_dictionary.csv` – columns from extraction stage  
- `processed_data_dictionary.csv` – final dataset columns  

---

## Known Limitations
- CourtListener’s `plain_text` is not always present; outcome labels may fall back to regex heuristics.  
- Model features are simple (court one-hot only); accuracy is not the focus.  
- API rate limits may throttle extraction if `max_cases` is large.  

---

## Branch Policy
- All development occurs in `dev`  
- Merge to `test` for integration checks  
- Merge `test` → `main` for final grading  
- All three branches are in sync for submission; grading happens on `main`  
