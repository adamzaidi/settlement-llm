# Court Opinion Outcome Triage — ETL + Baseline ML (CourtListener)

**Author:** Adam Zaidi  
**Last updated:** January 2026  

---

## Project Overview

This project builds a full **ETL + analytics pipeline** around U.S. court opinions using the **CourtListener API**.

The pipeline is designed as a **decision-support tool**, not just a classifier. It performs three core functions end-to-end:

1. **Collect** court opinions for a query (e.g., `corporation`) and enrich them with court and citation metadata  
2. **Extract outcome signals** from opinion text using **rule-based labeling with confidence scoring**  
3. **Triage uncertain cases** into a **human-in-the-loop review queue**, while training baseline ML models for comparison  

The project answers the following question:

> **How do outcomes vary across courts, and which cases require human review because outcome extraction is uncertain?**

---

## Scope and Design Philosophy

### What This Project Is

- A reproducible **ETL + analytics pipeline**
- Focused on **opinion-level dispositions**, not docket-level procedural status
- Explicitly uncertainty-aware via confidence scores and review flags
- Designed with auditability, explainability, and extensibility in mind

### What This Project Is Not

- A production-grade legal outcome predictor
- A substitute for docket metadata or PACER data
- An attempt to infer the full “true case result” beyond the opinion itself

---

## Outcome Labels

This project classifies **what the court did in the opinion**, not the full lifecycle outcome of the case.

### Coarse Outcome (`outcome_code`)

- **0 — other / unclear**
- **1 — affirmed_or_dismissed**
- **2 — changed_or_mixed**  
  (reversed, vacated, remanded, or mixed outcomes)

### Fine Outcome (`outcome_label_fine`, `outcome_code_fine`)

- `affirmed`
- `dismissed`
- `reversed`
- `vacated`
- `remanded`
- `mixed`
- `other`

Each labeled opinion includes:
- An **evidence snippet**
- A **confidence score** (heuristic, 0–1)
- A **needs_review flag** indicating whether human review is recommended

---

## Human-in-the-Loop Review

Uncertainty is handled explicitly rather than hidden.

Cases are flagged for review when:
- The outcome is `other` or `mixed`
- The confidence score falls below a threshold (default: `0.60`)
- Strong disposition language is missing or ambiguous

Flagged cases are written to:

```
data/processed/review_queue.csv
```

This design allows the pipeline to function as a **triage tool**, not just a classifier.

---

## How to Run the Pipeline

### 1. Clone the repository and install dependencies

```bash
git clone <your_repo_url>
cd <repo_name>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file or export manually:

```bash
COURTLISTENER_API_KEY=YOUR_API_KEY
COURTLISTENER_EMAIL=your@email.com
```

### 3. Run the full pipeline

```bash
python main.py run
```

---

## Pipeline Stages

### 1. Extract
- Source: CourtListener `/search/` and `/opinions/` endpoints  
- Enrichment includes:
  - Court metadata
  - Citation data
  - Opinion text signals (when available)

**Output**
```
data/extracted/raw_data.csv
```

---

### 2. Transform
- Normalizes types and timestamps
- Cleans court and citation fields
- Extracts opinion text signals
- Applies rule-based outcome labeling
- Computes:
  - Confidence score
  - Disposition-zone indicators
  - Review flags

**Outputs**
```
data/processed/processed_data.csv
data/processed/review_queue.csv
```

---

### 3. Load
- Loads processed data into a pandas DataFrame
- Performs lightweight schema and quality validation

---

### 4. Modeling
- Baseline classifiers:
  - Logistic Regression
  - Random Forest
- Features:
  - One-hot encoded court
- Labels:
  - Coarse (`outcome_code`)
  - Fine (`outcome_code_fine`)
- Training skips gracefully if only one class is present

**Outputs**
```
data/model-eval/
├── classification_report_*.csv
├── confusion_*.csv
└── evaluation_summary.json
```

---

### 5. Visualizations

Generates report-ready charts, including:
- Outcome distributions
- Top courts by volume
- Court Pareto analysis
- Outcomes over time
- Opinion word count distributions

**Outputs**
```
data/outputs/*.png
```

---

## Validation and Logging

- All pipeline steps log to:
```
logs/pipeline.log
```
- Schema and domain validation occurs after loading
- Model training always writes evaluation artifacts, even if training is skipped

---

## Repository Structure

```
.
├── etl/                # Extract / Transform / Load
├── analysis/           # Modeling and evaluation
├── vis/                # Visualization generation
├── utils/              # Logging, validation, helpers
├── data/
│   ├── extracted/
│   ├── processed/
│   ├── model-eval/
│   ├── outputs/
│   └── reference-tables/
├── runs/               # Run artifacts (params, logs, plots)
├── logs/
├── main.py             # Pipeline runner
└── README.md
```

---

## Data Dictionaries

Reference tables are located in:

```
data/reference-tables/
```

They document:
- Raw extraction fields
- Processed dataset columns
- Outcome codes and meanings

---

## Known Limitations

- Opinion text does not always contain explicit disposition language
- CourtListener `plain_text` availability is inconsistent
- Outcome labeling relies on heuristics, not authoritative docket status
- Models use intentionally simple features (court only)

---

## Future Extensions

- SQL-backed storage (SQLite or Postgres)
- Docket-level enrichment
- Embedding-based outcome classification
- Active learning using review queue feedback
- Court-specific disposition modeling

---

All branches are kept in sync for reproducibility.
