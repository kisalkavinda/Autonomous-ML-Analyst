# 🤖 Autonomous ML Analyst

> **An intelligent, defensive, and autonomous machine learning system that takes a raw CSV and outputs a production-grade model, report, and deployment package.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Project Overview

The **Autonomous ML Analyst** is a "data-scientist-in-a-box" application. Unlike standard AutoML tools that simply brute-force models, this system emphasizes **defensive engineering** and **autonomous decision-making**. It mimics the workflow of a senior data scientist: identifying data quality issues, cleaning "dirty" targets, handling missing data, selecting the correct task (Regression vs. Classification), and generating a professional audit report.

It is designed to be **crash-proof**, handling edge cases like empty features, headerless files, and corrupted targets gracefully.

---

## 🌟 Features

### 🔬 Lab Mode (Training)

| Feature | Description |
|---|---|
| **Smart Target Selection** | Auto-detects the most likely target column using keyword matching, negative keywords (ID, Date, Year), and backward scanning heuristics. |
| **Headerless CSV Support** | Toggle to auto-generate column names (`Feature_0`, `Feature_1`, ...) for CSVs without headers. |
| **Currency & Date Auto-Cleaning** | Detects and strips currency symbols (`$`, `£`, `€`, `¥`) and parses date columns into Month/Quarter/Year features. |
| **Dirty Target Coercion** | Automatically cleans mixed text/number targets (e.g., "Cyberpunk" in a Sales column) by coercing invalid values to NaN. |
| **Data Leakage Detection** | Two-layer protection: Semantic mapping (e.g., drops "Profit" when predicting "Revenue") + Correlation threshold (drops features with >0.95 correlation to target). |
| **Numeric ID Detection** | Identifies and drops columns that are 100% unique (like `PassengerId`) to prevent overfitting. |
| **Feature Engineering** | Auto-generates derived features: Log transforms (for skewed data), Polynomial features, and Interaction terms between top correlated features. |
| **SMOTE for Imbalanced Data** | Detects class imbalance and applies SMOTE (Synthetic Minority Over-sampling) to balance the dataset. |
| **Class Weight Balancing** | Applies `class_weight='balanced'` to models when imbalance is detected, with a fallback for very small datasets (<50 samples). |
| **Hyperparameter Tuning** | Uses `RandomizedSearchCV` to tune the champion model's hyperparameters for optimal performance. |
| **Cross-Validation** | Automatically uses Stratified K-Fold CV for small datasets to ensure robust evaluation. |
| **Model Tournament** | Trains multiple candidate models (RandomForest, GradientBoosting, Linear/Logistic, etc.) and selects the champion based on MAE (Regression) or F1-Macro (Classification). |
| **Permutation Importance** | Model-agnostic feature importance using `sklearn.inspection.permutation_importance`, ensuring interpretability for all model types including ensembles. |
| **Model Persistence** | Saves the trained pipeline as `best_model.pkl` with metadata (`best_model_metadata.json`) for later use. |
| **Memory Cleanup** | Automatically frees training data from memory after training completes using `gc.collect()` to prevent OOM errors. |

### 🏭 Factory Mode (Inference / Worker Mode)

| Feature | Description |
|---|---|
| **Deterministic Feature Engineering** | Replays the exact transformation logic (Log transforms, Poly features) used during training to ensure prediction consistency. |
| **Smart Validation** | Automatically detects data drift, missing columns, type mismatches, and imputes missing values with informational feedback. |
| **Confidence Intervals** | For regression tasks using Random Forest models, calculates variance-based 95% confidence intervals from individual decision trees (μ ± 1.96σ). |
| **Download Predictions** | Export predictions as a downloadable CSV file. |

### 🚀 Auto-Generated FastAPI Deployment

The system packages your trained model into a ready-to-deploy ZIP file containing:
- `main.py`: A FastAPI server with auto-generated Pydantic schemas matching your dataset's feature types.
- `Dockerfile`: Container configuration for instant deployment.
- `requirements.txt`: Minimal dependencies for the inference environment.
- `metadata.json`: Model versioning and training context.
- `README.md`: Step-by-step deployment instructions.

### 📊 Professional Reporting

The Analysis tab auto-generates a Markdown report containing:
- **Dataset Overview**: Row/column counts, numeric vs. categorical features, target distribution.
- **Executive Summary**: Best model, target column, task type.
- **Validation Notes**: All quality checks passed.
- **Data Quality Interventions**: Dropped columns, warnings, and leakage detection results.
- **Preprocessing Strategy**: Imputation methods, scaling, encoding decisions.
- **Model Leaderboard**: All trained models ranked by performance metric.
- **Feature Importance Chart**: Interactive bar chart of top feature drivers.

### 🎨 Premium UI

- **Dark Theme**: Developer-centric "Neon Blue" dark theme configured via `.streamlit/config.toml`.
- **Large, Accessible Fonts**: All text, tabs, buttons, tables, and reports use increased font sizes for clear visibility.
- **Reset Button**: A sidebar "🔄 Reset App" button to clear all session state and free memory instantly.
- **Neon Glow Buttons**: Custom CSS with hover animations and gradient headers.

---

## 🏗️ System Architecture

### Pipeline Flow

```
User Upload → Validator → Preprocessor → Feature Engineer → Model Tournament → Reporter
     ↓              ↓           ↓               ↓                  ↓              ↓
   app.py    data_validator  preprocessing  preprocessing    model_trainer   report_generator
                  .py            .py            .py              .py              .py
```

### Directory Structure

```text
Autonomous-ML-Analyst/
├── app.py                      # Main Streamlit Application (UI + Orchestration)
├── src/                        # Core Engine
│   ├── config.py               # Configuration & Thresholds (via .env)
│   ├── state.py                # AnalysisState (Audit Log Dataclass)
│   ├── exceptions.py           # Custom Exceptions (InsufficientDataError, etc.)
│   ├── data_validator.py       # Data Quality: Leakage, ID Detection, Target Cleaning
│   ├── preprocessing.py        # Feature Engineering + Sklearn Pipelines
│   ├── model_trainer.py        # Model Tournament, Tuning & Confidence Intervals
│   ├── report_generator.py     # Markdown Report Engine
│   └── utils.py                # Utilities: Dataset Cleaning, Model Saving
├── tests/                      # Verification Suite
│   ├── test_validator.py       # Unit Tests: Data Validator
│   ├── test_preprocessing.py   # Unit Tests: Preprocessing
│   ├── test_model_trainer.py   # Unit Tests: Model Trainer
│   ├── test_report_generator.py# Unit Tests: Report Generator
│   ├── verify_happy_path.py    # Integration: Full Pipeline
│   ├── verify_tiny_dataset.py  # Edge Case: Small Datasets
│   ├── verify_id_trap.py       # Edge Case: ID Column Traps
│   ├── verify_dirty_target.py  # Edge Case: Mixed-Type Targets
│   ├── verify_no_features.py   # Edge Case: All Features Dropped
│   ├── verify_cleaning.py      # Edge Case: Currency/Date Cleaning
│   ├── verify_improvements.py  # Verification: All v1.1 Features
│   └── verify_tuning.py        # Verification: Hyperparameter Tuning
├── .streamlit/
│   └── config.toml             # Streamlit Theme Configuration
├── .env                        # Environment Variables (Thresholds)
├── .env.example                # Template for .env
├── requirements.txt            # Python Dependencies
└── README.md                   # This File
```

### Key Components

| Module | Role | Key Functions |
|---|---|---|
| `config.py` | Centralized `.env`-driven thresholds | `AnalysisConfig` (max_missing, max_cardinality, max_correlation) |
| `state.py` | Audit trail dataclass | `AnalysisState` (dropped_columns, warnings, model_scores, feature_importance) |
| `exceptions.py` | Custom error hierarchy | `InsufficientDataError`, `TargetConstantError`, `ExcessiveMissingDataError` |
| `data_validator.py` | Data quality gatekeeper | `validate_dataset()`, `suggest_target_column()`, Semantic Leakage Map |
| `preprocessing.py` | Feature engineering + pipelines | `engineer_features()`, `build_preprocessor()`, `detect_imbalance()` |
| `model_trainer.py` | ML tournament engine | `run_experiment()`, `predict_with_confidence()` |
| `report_generator.py` | Markdown reporting | `generate_markdown_report()` |
| `utils.py` | Utilities | `clean_dataset()`, `save_model_package()` |

---

## 🚀 How It Works

### Lab Mode (Training)

1.  **Upload**: User uploads a CSV file.
2.  **Target Selection**: System auto-suggests the target column using smart heuristics. User can override.
3.  **Validation**:
    *   Checks for headers (user toggle), minimum samples, constant targets.
    *   Drops rows with missing/corrupted targets.
    *   Drops 100% null columns, high-cardinality IDs, and numeric identifiers.
    *   Runs semantic leakage detection (keyword mapping) and correlation-based leakage detection.
4.  **Feature Engineering**:
    *   Detects skewed features and applies Log transforms.
    *   Generates Polynomial and Interaction features from top correlated columns.
    *   Stores metadata for deterministic replay in Factory mode.
5.  **Preprocessing**:
    *   Builds a Scikit-Learn `ColumnTransformer` dynamically.
    *   Numeric: Impute (Median) + StandardScaler.
    *   Categorical: Impute (Most Frequent) + OneHotEncoder.
6.  **Training**:
    *   Detects task type (Regression vs. Classification) based on target uniqueness.
    *   Detects class imbalance and applies SMOTE + class weight balancing.
    *   Trains 3-4 diverse candidate models in a tournament.
    *   Tunes the champion model with `RandomizedSearchCV`.
    *   Uses Cross-Validation for small datasets.
7.  **Reporting**: Generates a comprehensive Markdown report with dataset stats, model leaderboard, and feature importance chart.
8.  **Persistence**: Saves `best_model.pkl` and `metadata.json`.
9.  **Cleanup**: Automatically frees training data from memory.

### Factory Mode (Inference)

1.  **Switch to Factory**: Click the "Factory" tab (unlocked after training).
2.  **Upload Inference Data**: Upload a new CSV (must match training schema minus the target).
3.  **Auto-Validation**: Checks for missing columns, data types, null values, and data drift.
4.  **Feature Engineering Replay**: Applies the same transformations used during training.
5.  **Generate Predictions**: Download a CSV with predictions (includes confidence intervals for regression).

---

## 💻 Installation & Usage

### Prerequisites
*   Python 3.10+
*   Virtual Environment (recommended)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kisalkavinda/Autonomous-ML-Analyst.git
    cd Autonomous-ML-Analyst
    ```

2.  **Create and activate virtual environment:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment (Optional):**
    Copy `.env.example` to `.env` to tweak validation thresholds.
    ```bash
    cp .env.example .env
    ```

    Available thresholds:
    | Variable | Default | Description |
    |---|---|---|
    | `MAX_MISSING_PERCENTAGE` | `0.40` | Max % of missing values allowed in target |
    | `MAX_CARDINALITY_RATIO` | `0.90` | Max unique-value ratio for categorical columns |
    | `MIN_SAMPLES_ABSOLUTE` | `30` | Minimum rows required to train |
    | `MIN_SAMPLES_FOR_SPLIT` | `50` | Minimum rows for train/test split (else use CV) |
    | `MAX_CORRELATION` | `0.95` | Correlation threshold for leakage detection |

### Running the App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 🧪 Testing

The project includes a full suite of **unit tests** and **verification scripts** for edge cases.

**Run Unit Tests:**
```bash
pytest
```

**Run Verification Scripts:**
```bash
python tests/verify_happy_path.py          # Full pipeline happy path
python tests/verify_tiny_dataset.py        # Test size < limit
python tests/verify_id_trap.py             # Test ID column dropping
python tests/verify_dirty_target.py        # Test mixed-type targets
python tests/verify_no_features.py         # Test empty feature set crash
python tests/verify_cleaning.py            # Test currency/date cleaning
python tests/verify_improvements.py        # Test v1.1 features (SMOTE, tuning, etc.)
python tests/verify_tuning.py              # Test hyperparameter tuning
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Reactive UI and deployment |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML pipelines, models, preprocessing |
| `imbalanced-learn` | SMOTE for class imbalance |
| `shap` | Model explainability |
| `scipy` | Statistical functions (skewness detection) |
| `matplotlib` / `seaborn` | Visualization |
| `joblib` | Model serialization |
| `python-dotenv` | Environment variable management |
| `pytest` | Testing framework |

---

## 🛡️ Defensive Engineering

The system is designed to **never crash** on user input:

| Defense | Implementation |
|---|---|
| **Custom Exceptions** | `InsufficientDataError`, `TargetConstantError`, `ExcessiveMissingDataError` |
| **Global Error Handler** | Safety net in `app.py` ensures users never see raw Python tracebacks |
| **Leakage Detection** | Semantic mapping + correlation threshold prevents "too good to be true" models |
| **ID Trap Prevention** | Auto-drops 100% unique numeric/string columns |
| **Empty Feature Guard** | Prevents training if all features are dropped |
| **Memory Management** | Auto-GC after training + Reset button to clear session state |

---

**Author**: Kisal Kavinda  
**Status**: Production-Ready (v1.1)  
**Live Demo**: [Streamlit Cloud](https://autonomous-ml-analyst.streamlit.app/)
