# 🍬 11 Candy — Factory Reallocation & Shipping Optimizer

A Streamlit dashboard for predictive factory optimization and shipping efficiency analysis.

## Project Structure

```
11candy_project/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── data/
│   └── Nassau_Candy_Distributor.csv
├── models/
│   └── predictor.py        # ML prediction & recommendation engine
├── utils/
│   ├── data_loader.py      # Data loading & lead time computation
│   └── factory_info.py     # Factory coordinates, colors, mappings
└── .vscode/
    ├── settings.json       # Python formatting & linting settings
    ├── launch.json         # One-click Streamlit run config
    └── extensions.json     # Recommended extensions
```

## Quick Start

### 1. Open in VS Code
```bash
code 11candy_project/
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app

**Option A — Terminal:**
```bash
streamlit run app.py
```

**Option B — VS Code:**
Press `F5` or go to **Run → Start Debugging** → select **"Run Streamlit App"**

The app opens at **http://localhost:8501**

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🏭 Factory Simulator | Predict lead time per product × factory combination |
| 🔀 Scenario Analysis | Side-by-side current vs. reassigned factory comparison |
| 🏆 Recommendations | Ranked reassignment suggestions with risk scoring |
| 📊 Analytics | Sales, margin, lead time charts by region/product |

## Models Used

- **Random Forest Regressor** — primary lead time predictor (MAE, RMSE, R²)
- **Linear Regression** — baseline comparison
- **Gradient Boosting** — available in `models/predictor.py` for extension

## KPIs Tracked

- Lead Time Reduction (%)
- Profit Impact Stability
- Scenario Confidence Score
- Recommendation Coverage
