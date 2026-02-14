# 🤖 Autonomous ML Analyst

> **An intelligent, defensive, and autonomous machine learning system that takes a raw CSV and outputs a production-grade model and report.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Project Overview

The **Autonomous ML Analyst** is a "data-scientist-in-a-box" application. Unlike standard AutoML tools that simply brute-force models, this system emphasizes **defensive engineering** and **autonomous decision-making**. It mimics the workflow of a senior data scientist: identifying data quality issues, cleaning "dirty" targets, handling missing data, selecting the correct task (Regression vs. Classification), and generating a professional audit report.

It is designed to be **crash-proof**, handling edge cases like empty features, headerless files, and corrupted targets gracefully.

## 🌟 Advanced Features (v1.1)

### 1. Auto-Generated FastAPI Deployment
The system now includes a **"Production Deployment"** module in the Analysis tab. It automatically packages your trained model into a ready-to-deploy ZIP file containing:
- `main.py`: A high-performance FastAPI server.
- `Dockerfile`: Container configuration for instant deployment.
- `requirements.txt`: Minimal dependencies for the inference environment.
- `metadata.json`: Model versioning and training context.

### 2. Autonomous Inference Factory
The "Worker Mode" (Inference) has been hardened for production:
- **Deterministic Feature Engineering**: Replays the exact transformation logic (e.g., Log transforms, Poly features) used during training to ensure consistency.
- **Smart Validation**: Automatically detects data drift and imputes missing values with friendly, informational user feedback.

### 3. Real Confidence Intervals
For regression tasks using Random Forest models, the system calculates **variance-based uncertainty**. Instead of a single point prediction, you get a 95% confidence interval derived from the spread of individual decision trees ($\mu \pm 1.96\sigma$).

### 4. Premium UI
A developer-centric "Neon Blue" dark theme (configured via `.streamlit/config.toml`) provides a modern, high-contrast interface for long coding sessions.

---

## 🏗️ System Architecture

The system follows a modular, pipeline-driven architecture designed for maintainability and separation of concerns.

### System Flow

1. **User Upload** ➔ 📂 `app.py` receives CSV
2. **Validator** ➔ 🛡️ `data_validator.py` checks quality
   - *Pass:* Proceed
   - *Fail:* Raise Custom Error (e.g., `InsufficientDataError`)
3. **Preprocessor** ➔ ⚙️ `preprocessing.py` builds pipeline
   - *Numeric:* Impute (Median) + Scale
   - *Categorical:* Impute (Frequent) + OneHot
4. **Trainer** ➔ 🧠 `model_trainer.py` runs tournament
   - *Reg:* RF, GradientBoost, Linear
   - *Clf:* RF, GradientBoost, Logistic
5. **Reporter** ➔ 📄 `report_generator.py` writes Markdown

### Directory Structure

```text
Autonomous-ML-Analyst/
├── app.py                  # Main Streamlit Application (The "Face")
├── src/                    # The "Brain" (Core Engine)
│   ├── config.py           # Configuration & Settings
│   ├── state.py            # AnalysisState (Audit Log)
│   ├── data_validator.py   # Data Quality Guardrails
│   ├── preprocessing.py    # Dynamic Scikit-Learn Pipelines
│   ├── model_trainer.py    # Model Tournament & Selection
│   └── report_generator.py # Markdown Report Engine
├── tests/                  # Verification Suite
│   ├── verify_*.py         # "Break-It" Challenge Scripts
│   └── test_*.py           # Unit Tests
└── requirements.txt        # Dependencies
```

### Key Components

1.  **`src/config.py`**: Centralized configuration using `.env` for thresholds (e.g., `MAX_MISSING_PERCENTAGE`).
2.  **`src/state.py`**: A centralized `AnalysisState` dataclass that acts as the "brain," recording every decision, dropped column, and model score for the final report.
3.  **`src/data_validator.py` (The Bouncer)**:
    *   **Defensive Checks**: fast-fails on insufficient data or constant targets.
    *   **Auto-Cleaning**: Coerces "dirty" targets (mixed text/numbers), drops 100% null columns, and removes unique identifiers (IDs) to prevent overfitting.
4.  **`src/preprocessing.py`**:
    *   **Dynamic Pipeline**: Automatically detects numeric vs. categorical features.
    *   **Smart Imputation**: Uses Median for numbers and Most Frequent for categories.
5.  **`src/model_trainer.py`**:
    *   **Task Detection**: Infers Regression vs. Classification based on target uniqueness.
    *   **Tournament**: Trains multiple candidate models (RandomForest, GradientBoosting, Linear/Logistic) and selects the champion based on robust metrics (MAE or F1-Macro).

---

## 🛠️ Technical Implementation

### Technologies Used
*   **Python**: Core language.
*   **Streamlit**: Reactive UI for real-time feedback and report rendering.
*   **Pandas & NumPy**: High-performance data manipulation.
*   **Scikit-Learn**: The engine for pipelines, transformers, and model training.
*   **Pytest**: Comprehensive testing suite (Component & Integration tests).

### Engineering Highlights
*   **Defensive Programming**: Extensive use of custom exceptions (`InsufficientDataError`, `ExcessiveMissingDataError`) to handle failures gracefully.
*   **Global Error Handling**: A safety net in `app.py` ensures the user never sees a raw Python traceback.
*   **Robustness Fixes (Phase 6)**:
    *   **Headerless CSV Handling**: UI toggle to auto-generate column names.
    *   **Dirty Target Coercion**: Automatically cleans text values (e.g., "Cyberpunk" in a Sales column) to `NaN`.
    *   **Numeric ID Detection**: Identifies and drops columns that are 100% unique (like `PassengerId`).
    *   **Empty Feature Guard**: Prevents training if all features are dropped, providing a clear error message.

---

## 🚀 How It Works (The Workflow)

1.  **Upload**: User uploads a CSV file.
2.  **Validation**:
    *   Checks if the file has headers (user toggle).
    *   Drops rows where the target is missing or corrupted.
    *   Drops columns that are 100% null or high-cardinality IDs.
3.  **Preprocessing**:
    *   Builds a Scikit-Learn `ColumnTransformer`.
    *   Scales numeric data (StandardScaler).
    *   One-hot encodes categorical data.
4.  **Training**:
    *   Splits data into Train/Test sets (or uses Cross-Validation for small data).
    *   Trains 3-4 diverse models.
    *   Compares performance on the held-out test set.
5.  **Reporting**:
    *   Generates a Markdown report summarizing the Best Model, Data Quality interventions, and Feature Engineering steps.

---

### 6. Worker Mode (Factory) 🏭
Once a model is trained, the **Factory** tab unlocks. This is your production inference engine.
1.  **Switch to Factory**: Click the "Factory" tab (only visible after training).
2.  **Upload Inference Data**: Upload a new CSV (must match training schema minus the target).
3.  **Auto-Validation**: The system checks for missing columns, data types, and drift.
4.  **Generate Predictions**: Click predictions to get a downloadable CSV with results.
    *   *Includes confidence intervals for Regression tasks!*

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

### Running the App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 🧪 Testing

The project includes a full suite of unit tests and verification scripts for edge cases.

**Run Unit Tests:**
```bash
pytest
```

**Run "Break-It" Verification Scripts:**
```bash
python tests/verify_tiny_dataset.py       # Test size < limit
python tests/verify_id_trap.py            # Test ID dropping
python tests/verify_dirty_target.py       # Test mixed-type targets
python tests/verify_no_features.py        # Test empty feature set crash
```

---

**Author**: Kisal Kavinda
**Status**: Production-Ready (v1.0)
