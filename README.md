# 📊 Sales Data Mining

**Automated Data Preprocessing and Exploratory Analysis of Sales Data Using Data Mining**

Mini Project — Unit II: Data Mining (Introduction)

Upload any CSV file. The app reads it, cleans it, transforms it, draws six
interactive charts, and compares records — all in the browser.

---

## Navigation

```
🏠 Home                 upload CSV, summary metric cards
📋 Dataset Overview     preview, column info, statistical description
🧹 Data Preprocessing   cleaning, transformation, discretization, reduction
📊 Visualization        the six charts
📐 Similarity Analysis  Euclidean, Manhattan, Cosine
```

The sidebar always carries **Download Cleaned Dataset** and **Reset to Original**.

---

## The Six Charts

| # | Chart | Type | Columns the teacher selects |
|---|---|---|---|
| 1 | Sales by Category | Bar | category column, measure column, aggregation |
| 2 | Sales by Region | Bar | region column, measure column, vertical/horizontal |
| 3 | Monthly Sales | Line | date column, measure column, Month/Quarter/Year |
| 4 | Sales Distribution | Histogram | numerical column, number of bins |
| 5 | Sales vs Profit | Scatter | X column, Y column, colour-by column |
| 6 | Correlation | Heatmap | which numerical columns to include |

Every dropdown is filled automatically from the uploaded file. If a column is
named Sales / Revenue / Category / Region / Profit it is pre-selected, but the
app works perfectly with completely different names — it just changes which
option starts selected.

---

## Technology Stack

Python · Pandas · NumPy · Scikit-learn · Plotly · Streamlit

---

## Folder Structure

```
sales-data-mining/
├── app.py                      # the whole application
├── requirements.txt
├── README.md
└── data/
    ├── sales_dataset.csv       # sample: 10,120 rows x 13 columns
    ├── test_dataset_2.csv      # different column names, for testing
    └── generate_dataset.py     # script that built the sample data
```

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Deploy on Streamlit Community Cloud

**1. Create the GitHub repository**
Sign in at github.com → **New repository** → name it `sales-data-mining` →
set it **Public** → do not add a README → **Create repository**.

**2. Upload the files**

```bash
cd sales-data-mining
git init
git add .
git commit -m "Data mining mini project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sales-data-mining.git
git push -u origin main
```

Or use **Add file → Upload files** in the browser. Either way, confirm the
repository looks exactly like this:

```
app.py
requirements.txt
README.md
data/sales_dataset.csv
data/test_dataset_2.csv
data/generate_dataset.py
```

If `sales_dataset.csv` ends up in the root instead of inside `data/`, the
sample-dataset button will not work.

**3. Deploy**
Go to <https://share.streamlit.io> → **Sign in with GitHub** → **Create app**
→ **Deploy a public app from GitHub**.

| Field | Value |
|---|---|
| Repository | `YOUR_USERNAME/sales-data-mining` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL | any available name |

Click **Deploy**. First build takes 2–5 minutes. You get a URL like
`https://your-app-name.streamlit.app`.

**4. Test it**
Open the URL in an incognito window to confirm it works for a visitor who is
not signed in.

---

## Test Plan

| # | Action | Expected result |
|---|---|---|
| 1 | Click **Use Sample Sales Dataset** | 10,120 rows, 13 columns, 1,011 missing, 120 duplicates |
| 2 | Preprocessing → **Clean Dataset Automatically** | missing → 0, rows → 10,000 |
| 3 | Overview → Statistical Description | mean, median, mode, Q1, Q3, variance, IQR for 5 columns |
| 4 | Visualization → all six charts | every chart renders |
| 5 | Transformation → Min-Max | all values land between 0 and 1 |
| 6 | Discretization → quantile | Low/Medium/High ≈ 3,373 each |
| 7 | Reduction → 50% sample | 10,000 → 5,000 rows, means stay close |
| 8 | Similarity → rows 100 and 250 | three results + step-by-step table |
| 9 | Sidebar → Download Cleaned Dataset | CSV downloads with cleaning applied |
| 10 | Upload `test_dataset_2.csv` | Item_Type, Zone, Revenue, Margin auto-selected |
| 11 | Upload a text-only CSV | friendly warnings, no crash |
| 12 | Upload an empty CSV | "This file is empty…" message |

---

## Unit II Syllabus Mapping

| Syllabus topic | Where it is implemented |
|---|---|
| **Data Objects and Attribute Types** | `detect_column_types()` — Dataset Overview → Column Information |
| **Statistical Description of Data** | Dataset Overview → Statistical Description |
| **Data Cleaning** | Preprocessing → Cleaning (missing values, duplicates, type fix) |
| **Data Transformation** | Preprocessing → Transformation (Min-Max, Z-score) |
| **Data Discretization** | Preprocessing → Discretization (equal-frequency, equal-width, custom) |
| **Data Reduction** | Preprocessing → Reduction (column selection, sampling) |
| **Data Integration** | Future scope — out of scope for a single-CSV application |
| **Data Visualization** | Visualization page — six Plotly charts |
| **Similarity and Dissimilarity** | Similarity page — Euclidean, Manhattan, Cosine |

---

## Viva Notes

- **Why two DataFrames?** `df_original` never changes, `df_working` holds the
  cleaning. This is what makes before/after comparisons and the Reset button honest.
- **Why median instead of mean for filling?** The median is not dragged by
  outliers. The sample `Sales` column is strongly right-skewed.
- **Why standardize before computing distance?** `Unit_Price` reaches 95,000
  while `Discount` only reaches 0.3. Without scaling, `Unit_Price` alone would
  decide the distance.
- **Equal-width vs equal-frequency binning** on the sample `Sales` column:
  equal-frequency gives ~3,373 per bin, equal-width gives 9,590 / 461 / 69.
  The difference is caused by skew — a good thing to show live.
- **Why is date detection so cautious?** A column converts only if its *name*
  looks like a date **and** at least 80% of its values actually parse. A wrong
  guess would silently destroy the teacher's data.
