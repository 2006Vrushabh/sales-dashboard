"""
===============================================================================
  SALES DATA MINING  -  Automated Preprocessing, Analysis and Visualization
===============================================================================

  Mini Project  |  Unit II : Data Mining (Introduction)

  Upload any CSV file. The app will:
      1. Read and understand it          -> Data Objects & Attribute Types
      2. Clean it                        -> Data Pre-processing
      3. Transform and reduce it         -> Transformation & Discretization
      4. Draw 6 interactive charts       -> Data Visualization
      5. Compare any two records         -> Similarity & Dissimilarity

  KEY IDEA
  --------
  Two copies of the data are kept in memory:
      df_original -> exactly what was uploaded, never changed
      df_working  -> the cleaned version (this is what gets charted/downloaded)
  This makes every "before vs after" comparison honest, and lets the user
  reset at any time.

  Run:  streamlit run app.py
===============================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)

warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1 : CONFIGURATION, THEME AND CONSTANTS
# =============================================================================

st.set_page_config(
    page_title="Sales Data Mining",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "sales_dataset.csv"
)

# Colour palette used by every chart, so the whole app looks consistent.
PALETTE = ["#4C6FFF", "#00C2A8", "#FFB020", "#FF5C7C",
           "#8B5CF6", "#22C55E", "#F97316", "#06B6D4"]

# Name hints used ONLY to pre-select sensible defaults in the dropdowns.
# The app still works if none of these words appear in the column names.
HINTS_VALUE = ["sales", "revenue", "amount", "total"]
HINTS_PROFIT = ["profit", "margin", "income"]
HINTS_CATEGORY = ["category", "product", "segment", "type", "item"]
HINTS_REGION = ["region", "state", "city", "country", "zone", "area"]
HINTS_DATE = ["date", "time", "day", "month", "year", "timestamp"]

# A text column with more distinct values than this is treated as an ID and is
# not offered for grouping (for example Order_ID with 10,000 unique values).
MAX_GROUPS = 50


def inject_custom_style():
    """A small amount of CSS to make the app look polished."""
    st.markdown(
        """
        <style>
            /* Hero section without gradient banner */
            .hero {
                background: transparent;
                padding: 10px 0;
                border-radius: 0;
                color: inherit;
                margin-bottom: 22px;
            }

            .hero h1 {
                margin: 0;
                font-size: 30px;
                font-weight: 700;
                color: inherit;
            }

            .hero p {
                margin: 8px 0 0 0;
                font-size: 15px;
                opacity: 0.8;
                color: inherit;
            }

            /* Keep pills working */
            .pill {
                display: inline-block;
                background: rgba(76, 111, 255, 0.12);
                border-radius: 20px;
                padding: 3px 13px;
                font-size: 12.5px;
                margin: 10px 7px 0 0;
                color: inherit;
            }

            /* Section heading */
            .section-title {
                font-size: 20px;
                font-weight: 650;
                border-left: 5px solid #4C6FFF;
                padding-left: 12px;
                margin: 26px 0 14px 0;
            }

            /* Tabs */
            .stTabs [data-baseweb="tab"] {
                font-size: 15px;
                padding: 9px 18px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def hero(title, subtitle, pills=None):
    """Draw the gradient banner at the top of a page."""
    pill_html = ""
    if pills:
        pill_html = "".join(
            "<span class='pill'>" + str(p) + "</span>" for p in pills
        )
    st.markdown(
        "<div class='hero'><h1>" + title + "</h1><p>" + subtitle + "</p>"
        + pill_html + "</div>",
        unsafe_allow_html=True,
    )


def section(text):
    """Draw a section heading with a coloured left bar."""
    st.markdown("<div class='section-title'>" + text + "</div>",
                unsafe_allow_html=True)


def style_chart(figure, title):
    """Apply one consistent look to every Plotly chart in the app."""
    figure.update_layout(
        title=dict(text=title, font=dict(size=18)),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=58, b=12),
        height=430,
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    return figure


# =============================================================================
# SECTION 2 : HELPER FUNCTIONS
# =============================================================================

def require(condition, message, kind="warning"):
    """
    Guard clause used all over the app.

    Instead of letting Python raise an error (which would show an ugly red
    traceback to the teacher), we check a condition first. If it fails we show
    a friendly message and return False so the caller can simply stop.
    """
    if condition:
        return True
    if kind == "error":
        st.error(message)
    elif kind == "info":
        st.info(message)
    else:
        st.warning(message)
    return False


def looks_like_a_date_column(series, column_name):
    """
    Decide whether a text column is really a date column.

    Deliberately CONSERVATIVE: we require BOTH
        (a) the column name contains a date-like word, AND
        (b) at least 80% of sampled values actually parse as dates.
    A wrong guess would silently damage the teacher's data.
    """
    if not any(hint in str(column_name).lower() for hint in HINTS_DATE):
        return False

    sample = series.dropna().astype(str).head(200)
    if len(sample) == 0:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
    except Exception:
        return False
    return parsed.notna().mean() >= 0.8


def auto_parse_date_columns(df):
    """Convert obvious date columns from text to real datetime values."""
    converted = []
    for column in df.columns:
        series = df[column]
        # "Text-like" means not numeric, not boolean, not already datetime.
        # We check it this way instead of comparing dtype to "object", because
        # newer pandas versions report text columns as "str" instead.
        is_text_like = not (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
        )
        if is_text_like and looks_like_a_date_column(series, column):
            try:
                df[column] = pd.to_datetime(df[column], errors="coerce")
                converted.append(column)
            except Exception:
                pass
    return df, converted


def detect_column_types(df):
    """
    Classify every column as numerical, categorical or datetime.

    This implements the Unit II topic "Data Objects and Attribute Types".
    Nothing is hardcoded to one dataset - the decision comes from the dtype.
    """
    numerical, categorical, datetimes = [], [], []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            datetimes.append(column)
        elif pd.api.types.is_bool_dtype(series):
            categorical.append(column)          # True/False acts like a category
        elif pd.api.types.is_numeric_dtype(series):
            numerical.append(column)
        else:
            categorical.append(column)
    return {"numerical": numerical, "categorical": categorical, "datetime": datetimes}


def groupable_columns(df, categorical_columns):
    """Keep only categorical columns with few enough categories to plot."""
    return [
        column for column in categorical_columns
        if 1 <= df[column].nunique(dropna=True) <= MAX_GROUPS
    ]


def guess_column(candidates, hints):
    """
    Pick the best default column from `candidates` using name hints.

    Example: guess_column(numerical, ["sales","revenue"]) returns the column
    named "Sales" if it exists. If nothing matches we return the first
    candidate, so the dropdown is never empty.
    """
    for hint in hints:
        for column in candidates:
            if hint in str(column).lower():
                return column
    return candidates[0] if candidates else None


def index_of(options, preferred):
    """Position of `preferred` inside `options`, or 0 if it is not there."""
    return options.index(preferred) if preferred in options else 0


def csv_bytes(df):
    """Convert a DataFrame to CSV bytes for the download button."""
    return df.to_csv(index=False).encode("utf-8")


def log(message):
    """Record a preprocessing step so it can be shown on the Home page."""
    st.session_state.action_log.append(message)


# =============================================================================
# SECTION 3 : LOADING DATA
# =============================================================================

def load_csv(uploaded_file):
    """
    Read an uploaded CSV safely.
    Returns (DataFrame, error_message) - exactly one of them is None.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        return None, "This file is empty. Please upload a CSV that contains data."
    except pd.errors.ParserError:
        return None, "This file could not be read as a CSV. Please check the format."
    except UnicodeDecodeError:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin-1")
        except Exception:
            return None, "The file encoding could not be read. Please save it as UTF-8 CSV."
    except Exception as error:
        return None, "The file could not be read. Details: " + str(error)

    if df is None or df.shape[0] == 0:
        return None, "The file was read successfully but contains no rows."
    if df.shape[1] == 0:
        return None, "The file was read successfully but contains no columns."
    return df, None


@st.cache_data
def load_sample_dataset():
    """Load the bundled sample dataset (cached so it is only read once)."""
    if not os.path.exists(SAMPLE_DATA_PATH):
        return None
    return pd.read_csv(SAMPLE_DATA_PATH)


def activate_dataset(df, source_label):
    """Store a freshly loaded dataset and clear any previous work."""
    df = df.copy()
    df, converted = auto_parse_date_columns(df)
    st.session_state.df_original = df.copy()
    st.session_state.df_working = df.copy()
    st.session_state.data_source = source_label
    st.session_state.action_log = []
    if converted:
        log("Detected and parsed date column(s): " + ", ".join(converted))


# Streamlit re-runs this whole file on every click, so session_state is the
# only way to remember the dataset between clicks.
for key, default in [
    ("df_original", None), ("df_working", None),
    ("data_source", None), ("action_log", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# =============================================================================
# SECTION 4 : PAGE 1  -  HOME
# =============================================================================

def page_home():
    hero(
        "📊 Sales Data Mining",
        "Automated Data Preprocessing and Exploratory Analysis of Sales Data",
        [ "Upload any CSV", "Clean → Transform → Visualize"],
    )

    section("1 · Load your dataset")

    left, right = st.columns([3, 2])

    with left:
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="Any comma-separated file with a header row in the first line.",
        )
        if st.button("🔍  Analyze Dataset", type="primary", width="stretch"):
            if uploaded_file is None:
                st.warning("Please choose a CSV file first, or load the sample dataset.")
            else:
                df, error = load_csv(uploaded_file)
                if error:
                    st.error(error)
                else:
                    activate_dataset(df, uploaded_file.name)
                    st.success("Loaded successfully: " + uploaded_file.name)

    with right:
        with st.container(border=True):
            st.markdown("**Supported format**")
            st.markdown(
                "- File type `.csv`\n"
                "- First row = column headers\n"
                "- Any number of rows and columns\n"
                "- Numbers, text and dates all supported\n"
                "- Missing values may be blank or `NA`"
            )
            if st.button("📁  Use Sample Sales Dataset", width="stretch"):
                sample = load_sample_dataset()
                if sample is None:
                    st.error("Sample file not found. Expected it at data/sales_dataset.csv")
                else:
                    activate_dataset(sample, "sales_dataset.csv (sample)")
                    st.success("Sample dataset loaded.")

    df = st.session_state.df_working
    if df is None:
        st.info("👆 Upload a CSV file or click **Use Sample Sales Dataset** to begin.")
        return

    # ---------------------- Summary metric cards -------------------------
    types = detect_column_types(df)
    section("2 · Dataset summary")
    st.caption("Currently loaded: " + str(st.session_state.data_source))

    row = st.columns(4)
    with row[0].container(border=True):
        st.metric("Rows (Data Objects)", f"{df.shape[0]:,}")
    with row[1].container(border=True):
        st.metric("Columns (Attributes)", f"{df.shape[1]:,}")
    with row[2].container(border=True):
        st.metric("Numerical Columns", len(types["numerical"]))
    with row[3].container(border=True):
        st.metric("Categorical Columns", len(types["categorical"]))

    row = st.columns(4)
    missing_total = int(df.isna().sum().sum())
    duplicate_total = int(df.duplicated().sum())
    cells = df.shape[0] * df.shape[1]
    completeness = 100 * (1 - missing_total / cells) if cells else 100.0

    with row[0].container(border=True):
        st.metric("Missing Values", f"{missing_total:,}")
    with row[1].container(border=True):
        st.metric("Duplicate Rows", f"{duplicate_total:,}")
    with row[2].container(border=True):
        st.metric("Data Completeness", f"{completeness:.2f} %")
    with row[3].container(border=True):
        st.metric("Date Columns", len(types["datetime"]))

    if missing_total or duplicate_total:
        st.warning(
            "This dataset needs cleaning: " + f"{missing_total:,}" + " missing value(s) and "
            + f"{duplicate_total:,}" + " duplicate row(s). Go to **🧹 Data Preprocessing**."
        )
    else:
        st.success("This dataset is already clean — no missing values and no duplicates.")

    section("3 · Quick preview")
    st.dataframe(df.head(8), width="stretch")

    if st.session_state.action_log:
        with st.expander("🧾  Steps applied so far"):
            for number, entry in enumerate(st.session_state.action_log, start=1):
                st.write(str(number) + ". " + entry)


# =============================================================================
# SECTION 5 : PAGE 2  -  DATASET OVERVIEW
# =============================================================================

def page_overview():
    hero(
        "📋 Dataset Overview",
        "Data objects, attribute types and the statistical description of the data",
        ["Data Objects & Attribute Types", "Statistical Description"],
    )

    df = st.session_state.df_working
    if not require(df is not None, "Please load a dataset on the Home page first.", "info"):
        return

    types = detect_column_types(df)
    tab_preview, tab_info, tab_stats = st.tabs(
        ["👁️  Preview", "🔎  Column Information", "📐  Statistical Description"]
    )

    # ------------------------------ PREVIEW ------------------------------
    with tab_preview:
        section("First 10 rows")
        st.dataframe(df.head(10), width="stretch")
        if st.checkbox("Show the last 10 rows as well"):
            section("Last 10 rows")
            st.dataframe(df.tail(10), width="stretch")

    # --------------------------- COLUMN INFO -----------------------------
    with tab_info:
        row = st.columns(3)
        with row[0].container(border=True):
            st.metric(" Numerical", len(types["numerical"]))
        with row[1].container(border=True):
            st.metric(" Categorical", len(types["categorical"]))
        with row[2].container(border=True):
            st.metric(" Date / Time", len(types["datetime"]))

        section("Every column at a glance")
        rows = []
        for column in df.columns:
            missing = int(df[column].isna().sum())
            if column in types["numerical"]:
                attribute_type = " Numerical"
            elif column in types["datetime"]:
                attribute_type = " Date / Time"
            else:
                attribute_type = " Categorical"
            rows.append({
                "Column": column,
                "Attribute Type": attribute_type,
                "Data Type": str(df[column].dtype),
                "Unique Values": int(df[column].nunique(dropna=True)),
                "Missing": missing,
                "Missing %": round(100 * missing / len(df), 2) if len(df) else 0,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        usable = groupable_columns(df, types["categorical"])
        if usable:
            section("Category frequency")
            chosen = st.selectbox("Choose a categorical column", usable)
            counts = df[chosen].value_counts(dropna=False).reset_index()
            counts.columns = [chosen, "Count"]
            counts["Percent"] = (100 * counts["Count"] / len(df)).round(2)

            left, right = st.columns([1, 2])
            with left:
                st.dataframe(counts, width="stretch", hide_index=True)
            with right:
                try:
                    figure = px.bar(counts, x=chosen, y="Count",
                                    color=chosen, color_discrete_sequence=PALETTE)
                    figure.update_layout(showlegend=False)
                    st.plotly_chart(style_chart(figure, "Frequency of " + chosen),
                                    width="stretch")
                except Exception:
                    st.caption("Chart could not be drawn for this column.")

    # ------------------------ STATISTICAL SUMMARY ------------------------
    with tab_stats:
        if not require(
            len(types["numerical"]) > 0,
            "Statistical description needs at least one numerical column. "
            "This dataset does not contain any.",
        ):
            return

        section("Statistical description of numerical attributes")
        stats_rows = []
        for column in types["numerical"]:
            series = df[column].dropna()
            if len(series) == 0:
                continue
            modes = series.mode()
            mode_value = modes.iloc[0] if not modes.empty else np.nan
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            stats_rows.append({
                "Attribute": column,
                "Count": int(series.count()),
                "Mean": round(float(series.mean()), 2),
                "Median": round(float(series.median()), 2),
                "Mode": round(float(mode_value), 2) if pd.notna(mode_value) else np.nan,
                "Min": round(float(series.min()), 2),
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "Max": round(float(series.max()), 2),
                "Std Dev": round(float(series.std()), 2),
                "Variance": round(float(series.var()), 2),
                "IQR": round(q3 - q1, 2),
            })

        if not stats_rows:
            st.warning("Numerical columns exist but contain no usable values.")
            return

        st.dataframe(pd.DataFrame(stats_rows), width="stretch", hide_index=True)

        with st.expander(" What do these measures mean?"):
            st.markdown(
                "- **Mean** — the average. Pulled around by extreme values.\n"
                "- **Median** — the middle value. A safer centre when outliers exist.\n"
                "- **Mode** — the most frequent value.\n"
                "- **Q1 / Q3** — the 25th and 75th percentiles.\n"
                "- **IQR = Q3 − Q1** — spread of the middle half of the data. "
                "Values outside `Q1 − 1.5×IQR` to `Q3 + 1.5×IQR` are outliers.\n"
                "- **Std Dev** — average distance of values from the mean.\n"
                "- **Variance** — the square of the standard deviation."
            )


# =============================================================================
# SECTION 6 : PAGE 3  -  DATA PREPROCESSING
# =============================================================================

def page_preprocessing():
    hero(
        "🧹 Data Preprocessing",
        "Cleaning, transformation, discretization and reduction of the dataset",
        ["Data Cleaning", "Transformation", "Discretization", "Reduction"],
    )

    df = st.session_state.df_working
    if not require(df is not None, "Please load a dataset on the Home page first.", "info"):
        return

    row = st.columns(4)
    with row[0].container(border=True):
        st.metric("Rows", f"{df.shape[0]:,}")
    with row[1].container(border=True):
        st.metric("Columns", f"{df.shape[1]:,}")
    with row[2].container(border=True):
        st.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    with row[3].container(border=True):
        st.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    tab_clean, tab_transform, tab_bin, tab_reduce = st.tabs(
        ["  Cleaning", "  Transformation", "  Discretization", "  Reduction"]
    )

    with tab_clean:
        tab_cleaning()
    with tab_transform:
        tab_transformation()
    with tab_bin:
        tab_discretization()
    with tab_reduce:
        tab_reduction()


# ----------------------------- A. CLEANING -----------------------------------

def tab_cleaning():
    df = st.session_state.df_working

    # ---- one-click option, because the teacher will like this ----
    section("Quick clean")
    st.caption(
        "Fills numerical gaps with the median, categorical gaps with the mode, "
        "and removes duplicate rows — all in one step."
    )
    if st.button("  Clean Dataset Automatically", type="primary", width="stretch"):
        types = detect_column_types(df)
        before_missing = int(df.isna().sum().sum())
        before_rows = df.shape[0]
        cleaned = df.copy()
        try:
            for column in types["numerical"]:
                if cleaned[column].isna().any():
                    cleaned[column] = cleaned[column].fillna(cleaned[column].median())
            for column in types["categorical"]:
                if cleaned[column].isna().any():
                    modes = cleaned[column].mode()
                    if not modes.empty:
                        cleaned[column] = cleaned[column].fillna(modes.iloc[0])
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        except Exception as error:
            st.error("Automatic cleaning failed. Details: " + str(error))
            return

        st.session_state.df_working = cleaned
        log("Automatic clean: median/mode imputation + duplicate removal.")
        st.success(
            "Done. Missing values " + f"{before_missing:,}" + " → "
            + f"{int(cleaned.isna().sum().sum()):,}" + "  |  Rows "
            + f"{before_rows:,}" + " → " + f"{cleaned.shape[0]:,}"
        )
        st.rerun()

    st.divider()

    # ---- missing values ----
    section("A · Missing values")
    missing_counts = df.isna().sum()
    if int(missing_counts.sum()) == 0:
        st.success("No missing values in the current dataset.")
    else:
        report = [
            {
                "Column": column,
                "Data Type": str(df[column].dtype),
                "Missing": int(missing_counts[column]),
                "Missing %": round(100 * missing_counts[column] / len(df), 2),
            }
            for column in df.columns if missing_counts[column] > 0
        ]
        st.dataframe(pd.DataFrame(report), width="stretch", hide_index=True)

        strategy = st.radio(
            "Treatment method",
            [
                "Fill numerical with MEDIAN and categorical with MODE (recommended)",
                "Fill numerical columns with MEAN",
                "Fill numerical columns with MEDIAN",
                "Fill categorical columns with MODE",
                "Remove all rows containing missing values",
            ],
        )
        st.caption(
            "Removing rows is simple but throws data away. Filling (imputation) "
            "keeps every row but slightly changes the distribution. The median "
            "is safer than the mean when outliers are present."
        )

        if st.button("  Apply treatment"):
            types = detect_column_types(df)
            before_missing = int(df.isna().sum().sum())
            before_rows = df.shape[0]
            cleaned = df.copy()
            try:
                if strategy.startswith("Remove"):
                    cleaned = cleaned.dropna().reset_index(drop=True)
                if "MEAN" in strategy:
                    for column in types["numerical"]:
                        cleaned[column] = cleaned[column].fillna(cleaned[column].mean())
                if "MEDIAN" in strategy:
                    for column in types["numerical"]:
                        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
                if "MODE" in strategy:
                    for column in types["categorical"]:
                        modes = cleaned[column].mode()
                        if not modes.empty:
                            cleaned[column] = cleaned[column].fillna(modes.iloc[0])
            except Exception as error:
                st.error("Could not apply this treatment. Details: " + str(error))
                return

            st.session_state.df_working = cleaned
            log("Missing values handled: " + strategy)
            after = st.columns(2)
            after[0].metric("Missing Values", f"{int(cleaned.isna().sum().sum()):,}",
                            delta=f"{int(cleaned.isna().sum().sum()) - before_missing:,}")
            after[1].metric("Rows", f"{cleaned.shape[0]:,}",
                            delta=f"{cleaned.shape[0] - before_rows:,}")
            st.success("Treatment applied.")

    st.divider()

    # ---- duplicates ----
    section("B · Duplicate rows")
    duplicate_mask = df.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count == 0:
        st.success("No duplicate rows in the current dataset.")
    else:
        st.warning(
            f"{duplicate_count:,}" + " duplicate row(s) found. Duplicates bias every "
            "statistical measure, so they should normally be removed."
        )
        with st.expander("👀  Preview the duplicate rows"):
            st.dataframe(df[duplicate_mask].head(15), width="stretch")
        if st.button("🗑️  Remove duplicates"):
            before_rows = df.shape[0]
            cleaned = df.drop_duplicates(keep="first").reset_index(drop=True)
            st.session_state.df_working = cleaned
            log("Removed " + str(before_rows - cleaned.shape[0]) + " duplicate row(s).")
            st.success(
                "Removed " + str(before_rows - cleaned.shape[0]) + " row(s):  "
                + f"{before_rows:,}" + " → " + f"{cleaned.shape[0]:,}"
            )

    st.divider()

    # ---- data type fix ----
    section("C · Fix a column's data type")
    st.caption(
        "Useful when a date column was stored as text — the Monthly Sales chart "
        "needs a real date column. The conversion is previewed before it is applied."
    )
    controls = st.columns(2)
    column_to_fix = controls[0].selectbox("Column", list(df.columns))
    target_type = controls[1].selectbox("Convert to", ["Date / Time", "Numeric", "Text"])

    series = df[column_to_fix]
    try:
        if target_type == "Numeric":
            preview = pd.to_numeric(series, errors="coerce")
        elif target_type == "Date / Time":
            preview = pd.to_datetime(series, errors="coerce")
        else:
            preview = series.astype(str)
    except Exception as error:
        st.error("This conversion is not possible. Details: " + str(error))
        return

    failed = max(int(preview.isna().sum() - series.isna().sum()), 0)
    if failed > 0:
        st.warning("Preview: " + f"{failed:,}" + " value(s) cannot be converted and "
                   "would become missing.")
    else:
        st.info("Preview: every value converts safely.")

    if st.button("🔄  Apply conversion"):
        updated = df.copy()
        updated[column_to_fix] = preview
        st.session_state.df_working = updated
        log("Converted '" + column_to_fix + "' to " + target_type + ".")
        st.success("Column '" + column_to_fix + "' converted to " + target_type + ".")


# -------------------------- B. TRANSFORMATION --------------------------------

def tab_transformation():
    df = st.session_state.df_working
    types = detect_column_types(df)
    numerical = types["numerical"]

    if not require(len(numerical) > 0,
                   "Transformation needs at least one numerical column."):
        return

    st.markdown(
        r"""
**Min-Max Normalization** rescales values into the range 0 to 1:
$$X' = \frac{X - X_{min}}{X_{max} - X_{min}}$$

**Z-Score Standardization** rescales values to mean 0 and standard deviation 1:
$$Z = \frac{X - \mu}{\sigma}$$

Both are needed because attributes on different scales (Quantity 1–15 vs
Unit_Price 51–94,968) cannot be compared fairly.
        """
    )

    selected = st.multiselect("Numerical column(s) to transform", numerical,
                              default=numerical[: min(3, len(numerical))])
    method = st.radio("Method", ["Min-Max Normalization", "Z-Score Standardization"],
                      horizontal=True)

    if not selected:
        st.info("Select at least one column.")
        return

    working = df[selected].copy()
    working = working.fillna(working.median(numeric_only=True))  # sklearn needs no NaN

    try:
        if method == "Min-Max Normalization":
            scaler, suffix = MinMaxScaler(), "_minmax"
        else:
            scaler, suffix = StandardScaler(), "_zscore"
        transformed = pd.DataFrame(
            scaler.fit_transform(working),
            columns=[column + suffix for column in selected],
            index=working.index,
        )
    except Exception as error:
        st.error("Transformation failed. Details: " + str(error))
        return

    left, right = st.columns(2)
    with left:
        st.markdown("**Before**")
        st.dataframe(working.head(8).round(2), width="stretch")
    with right:
        st.markdown("**After**")
        st.dataframe(transformed.head(8).round(4), width="stretch")

    st.markdown("**Effect on the summary statistics**")
    st.dataframe(
        pd.DataFrame(
            {
                "Before Min": working.min().round(2).values,
                "Before Max": working.max().round(2).values,
                "Before Mean": working.mean().round(2).values,
                "After Min": transformed.min().round(3).values,
                "After Max": transformed.max().round(3).values,
                "After Mean": transformed.mean().round(3).values,
            },
            index=selected,
        ),
        width="stretch",
    )

    if st.button("➕  Add transformed columns to the dataset"):
        updated = df.copy()
        for column in transformed.columns:
            updated[column] = transformed[column]
        st.session_state.df_working = updated
        log(method + " applied to: " + ", ".join(selected))
        st.success("Transformed columns added.")


# ------------------------- C. DISCRETIZATION ---------------------------------

def tab_discretization():
    df = st.session_state.df_working
    types = detect_column_types(df)
    numerical = types["numerical"]

    if not require(len(numerical) > 0,
                   "Discretization needs at least one numerical column."):
        return

    st.markdown(
        "**Discretization** converts a continuous numerical attribute into a few "
        "labelled intervals (bins) — for example Sales becomes Low / Medium / High, "
        "or Age becomes Young / Adult / Middle Age / Senior."
    )

    controls = st.columns(2)
    column = controls[0].selectbox("Numerical column to discretize", numerical)
    method = controls[1].radio(
        "Binning method",
        ["Equal-frequency (quantile)", "Equal-width", "Custom ranges (age-style)"],
    )

    try:
        if method.startswith("Equal-frequency"):
            # qcut puts an EQUAL NUMBER OF RECORDS into each bin
            binned = pd.qcut(df[column], q=3, labels=["Low", "Medium", "High"],
                             duplicates="drop")
            note = ("Quantile binning — each bin holds roughly the same number "
                    "of records.")
        elif method == "Equal-width":
            # cut splits the RANGE into equal-sized intervals
            binned = pd.cut(df[column], bins=3, labels=["Low", "Medium", "High"])
            note = ("Equal-width binning — the value range is split into three "
                    "equal intervals, so bins can hold very different counts.")
        else:
            edges_row = st.columns(4)
            b1 = edges_row[0].number_input("Boundary 1", value=25.0)
            b2 = edges_row[1].number_input("Boundary 2", value=40.0)
            b3 = edges_row[2].number_input("Boundary 3", value=60.0)
            labels_text = edges_row[3].text_input(
                "Labels (comma separated)", "Young,Adult,Middle Age,Senior")
            labels = [part.strip() for part in labels_text.split(",")]
            if len(labels) != 4:
                st.warning("Please give exactly 4 labels for the 4 intervals.")
                return
            if not (b1 < b2 < b3):
                st.warning("Boundaries must increase: Boundary 1 < 2 < 3.")
                return
            binned = pd.cut(df[column], bins=[-np.inf, b1, b2, b3, np.inf], labels=labels)
            note = "Custom ranges chosen manually."
    except ValueError as error:
        st.warning("This column cannot be split into 3 bins — it may have too few "
                   "distinct values. Details: " + str(error))
        return
    except Exception as error:
        st.error("Binning failed. Details: " + str(error))
        return

    st.info(note)
    counts = binned.value_counts(dropna=False).reset_index()
    counts.columns = ["Bin", "Records"]

    left, right = st.columns([1, 2])
    with left:
        st.dataframe(counts, width="stretch", hide_index=True)
    with right:
        try:
            figure = px.bar(counts, x="Bin", y="Records", color="Bin",
                            text="Records", color_discrete_sequence=PALETTE)
            figure.update_layout(showlegend=False)
            st.plotly_chart(style_chart(figure, "Records per bin — " + column),
                            width="stretch")
        except Exception:
            st.caption("Chart could not be drawn.")

    st.markdown("**Preview**")
    st.dataframe(
        pd.DataFrame({column: df[column].head(8), column + "_Bin": binned.head(8)}),
        width="stretch",
    )

    if st.button("➕  Add the bin column to the dataset"):
        updated = df.copy()
        updated[column + "_Bin"] = binned.astype(str)
        st.session_state.df_working = updated
        log("Discretized '" + column + "' using " + method + ".")
        st.success("Column '" + column + "_Bin' added.")


# ---------------------------- D. REDUCTION -----------------------------------

def tab_reduction():
    df = st.session_state.df_working
    st.markdown(
        "**Data reduction** produces a smaller dataset that still gives almost the "
        "same analytical result — by keeping fewer **columns** (attribute subset "
        "selection) or fewer **rows** (sampling)."
    )

    attribute_tab, sampling_tab = st.tabs(["Select columns", "Sample rows"])

    with attribute_tab:
        types = detect_column_types(df)
        suggested = (types["numerical"][:4] + types["categorical"][:2]) or list(df.columns)[:5]
        selected = st.multiselect("Columns to KEEP", list(df.columns), default=suggested)
        if not selected:
            st.info("Select at least one column.")
        else:
            row = st.columns(3)
            with row[0].container(border=True):
                st.metric("Original Columns", df.shape[1])
            with row[1].container(border=True):
                st.metric("Reduced Columns", len(selected))
            with row[2].container(border=True):
                st.metric("Reduction", f"{100 * (1 - len(selected) / df.shape[1]):.0f} %")
            st.dataframe(df[selected].head(8), width="stretch")
            if st.button("✂️  Apply column reduction"):
                st.session_state.df_working = df[selected].copy()
                log("Reduced dataset to " + str(len(selected)) + " column(s).")
                st.success("Dataset reduced.")

    with sampling_tab:
        percentage = st.slider("Percentage of rows to keep", 1, 100, 50)
        sample_size = max(1, int(len(df) * percentage / 100))
        try:
            sampled = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        except Exception as error:
            st.error("Sampling failed. Details: " + str(error))
            return

        row = st.columns(3)
        with row[0].container(border=True):
            st.metric("Original Rows", f"{len(df):,}")
        with row[1].container(border=True):
            st.metric("Sample Rows", f"{sample_size:,}")
        with row[2].container(border=True):
            st.metric("Kept", f"{percentage} %")

        types = detect_column_types(df)
        if types["numerical"]:
            compare_column = st.selectbox(
                "Check that the sample preserves the mean of", types["numerical"])
            means = st.columns(2)
            means[0].metric("Mean — full dataset", f"{df[compare_column].mean():,.2f}")
            means[1].metric("Mean — sample", f"{sampled[compare_column].mean():,.2f}")
            st.caption("Close means = a good sample. That is the whole point of "
                       "numerosity reduction.")

        if st.button("✂️  Apply sampling"):
            st.session_state.df_working = sampled
            log("Sampled " + str(percentage) + "% of rows.")
            st.success("Working dataset replaced with the sample.")


# =============================================================================
# SECTION 7 : PAGE 4  -  VISUALIZATION  (the six required charts)
# =============================================================================

def page_visualization():
    hero(
        "📊 Data Visualization",
        "Six interactive charts. Choose the columns from the dropdowns in each card.",
        ["Bar", "Bar", "Line", "Histogram", "Scatter", "Heatmap"],
    )

    df = st.session_state.df_working
    if not require(df is not None, "Please load a dataset on the Home page first.", "info"):
        return

    types = detect_column_types(df)
    numerical = types["numerical"]
    categorical = groupable_columns(df, types["categorical"])
    dates = types["datetime"]

    # Smart defaults, chosen from the column NAMES. If the names are different
    # the dropdowns still work - only the pre-selection changes.
    default_value = guess_column(numerical, HINTS_VALUE)
    default_profit = guess_column(
        [column for column in numerical if column != default_value], HINTS_PROFIT)
    default_category = guess_column(categorical, HINTS_CATEGORY)
    default_region = guess_column(categorical, HINTS_REGION)

    chart_1_sales_by_category(df, categorical, numerical, default_category, default_value)
    chart_2_sales_by_region(df, categorical, numerical, default_region, default_value)
    chart_3_monthly_sales(df, dates, numerical, default_value)
    chart_4_distribution(df, numerical, default_value)
    chart_5_sales_vs_profit(df, numerical, categorical, default_value, default_profit)
    chart_6_correlation(df, numerical)


def aggregate(df, group_column, value_column, how):
    """Group `value_column` by `group_column` using the chosen aggregation."""
    grouped = df.groupby(group_column, dropna=False)[value_column]
    result = {"Sum": grouped.sum, "Average": grouped.mean,
              "Count": grouped.count, "Maximum": grouped.max}[how]()
    result = result.reset_index()
    result.columns = [group_column, value_column]
    return result.sort_values(value_column, ascending=False)


# ------------------------- CHART 1 : SALES BY CATEGORY -----------------------

def chart_1_sales_by_category(df, categorical, numerical, default_category, default_value):
    with st.container(border=True):
        st.markdown("### 1️⃣  Sales by Category")
        st.caption("Bar chart — compares a measure across the categories of a column.")

        if not require(len(categorical) > 0 and len(numerical) > 0,
                       "This chart needs one categorical column and one numerical column."):
            return

        controls = st.columns(3)
        group_column = controls[0].selectbox(
            "Category column", categorical,
            index=index_of(categorical, default_category), key="c1_group")
        value_column = controls[1].selectbox(
            "Measure column", numerical,
            index=index_of(numerical, default_value), key="c1_value")
        how = controls[2].selectbox("Aggregation",
                                    ["Sum", "Average", "Count", "Maximum"], key="c1_agg")

        try:
            result = aggregate(df, group_column, value_column, how)
            figure = px.bar(result, x=group_column, y=value_column, color=group_column,
                            text_auto=".2s", color_discrete_sequence=PALETTE)
            figure.update_layout(showlegend=False)
            st.plotly_chart(
                style_chart(figure, how + " of " + value_column + " by " + group_column),
                width="stretch")
            with st.expander("View the numbers"):
                st.dataframe(result, width="stretch", hide_index=True)
        except Exception as error:
            st.error("This chart could not be drawn. Details: " + str(error))


# -------------------------- CHART 2 : SALES BY REGION ------------------------

def chart_2_sales_by_region(df, categorical, numerical, default_region, default_value):
    with st.container(border=True):
        st.markdown("### 2️⃣  Sales by Region")
        st.caption("Bar chart — the same idea applied to a location-type column.")

        if not require(len(categorical) > 0 and len(numerical) > 0,
                       "This chart needs one categorical column and one numerical column."):
            return

        controls = st.columns(3)
        group_column = controls[0].selectbox(
            "Region column", categorical,
            index=index_of(categorical, default_region), key="c2_group")
        value_column = controls[1].selectbox(
            "Measure column", numerical,
            index=index_of(numerical, default_value), key="c2_value")
        orientation = controls[2].selectbox("Layout", ["Vertical", "Horizontal"],
                                            key="c2_orient")

        try:
            result = aggregate(df, group_column, value_column, "Sum")
            if orientation == "Vertical":
                figure = px.bar(result, x=group_column, y=value_column, color=group_column,
                                text_auto=".2s", color_discrete_sequence=PALETTE)
            else:
                figure = px.bar(result, x=value_column, y=group_column, color=group_column,
                                orientation="h", text_auto=".2s",
                                color_discrete_sequence=PALETTE)
            figure.update_layout(showlegend=False)
            st.plotly_chart(
                style_chart(figure, "Total " + value_column + " by " + group_column),
                width="stretch")

            top = result.iloc[0]
            st.info("Highest: **" + str(top[group_column]) + "** with "
                    + f"{top[value_column]:,.2f}")
        except Exception as error:
            st.error("This chart could not be drawn. Details: " + str(error))


# -------------------------- CHART 3 : MONTHLY SALES --------------------------

def chart_3_monthly_sales(df, dates, numerical, default_value):
    with st.container(border=True):
        st.markdown("### 3️⃣  Monthly Sales")
        st.caption("Line chart — how a measure changes over time.")

        if not require(
            len(dates) > 0,
            "This chart needs a date column. None was detected. You can convert a "
            "text column to a date on the **Data Preprocessing → Cleaning** page.",
        ):
            return
        if not require(len(numerical) > 0, "This chart also needs a numerical column."):
            return

        controls = st.columns(3)
        date_column = controls[0].selectbox("Date column", dates, key="c3_date")
        value_column = controls[1].selectbox(
            "Measure column", numerical,
            index=index_of(numerical, default_value), key="c3_value")
        frequency = controls[2].selectbox("Group by", ["Month", "Quarter", "Year"],
                                          key="c3_freq")

        code = {"Month": "M", "Quarter": "Q", "Year": "Y"}[frequency]
        try:
            temporary = df[[date_column, value_column]].dropna()
            if temporary.empty:
                st.warning("No rows have both a valid date and a valid measure.")
                return
            temporary["Period"] = temporary[date_column].dt.to_period(code).astype(str)
            trend = temporary.groupby("Period")[value_column].sum().reset_index()
            trend = trend.sort_values("Period")

            figure = px.line(trend, x="Period", y=value_column, markers=True,
                             color_discrete_sequence=[PALETTE[0]])
            figure.update_traces(line=dict(width=3), marker=dict(size=7))
            st.plotly_chart(
                style_chart(figure, frequency + "ly trend of " + value_column),
                width="stretch")

            best = trend.loc[trend[value_column].idxmax()]
            st.info("Peak period: **" + str(best["Period"]) + "** with "
                    + f"{best[value_column]:,.2f}")
        except Exception as error:
            st.error("This chart could not be drawn. Details: " + str(error))


# ----------------------- CHART 4 : SALES DISTRIBUTION ------------------------

def chart_4_distribution(df, numerical, default_value):
    with st.container(border=True):
        st.markdown("### 4️⃣  Sales Distribution")
        st.caption("Histogram — how the values of one numerical column are spread out.")

        if not require(len(numerical) > 0, "This chart needs a numerical column."):
            return

        controls = st.columns(2)
        value_column = controls[0].selectbox(
            "Numerical column", numerical,
            index=index_of(numerical, default_value), key="c4_value")
        bins = controls[1].slider("Number of bins", 5, 100, 30, key="c4_bins")

        try:
            figure = px.histogram(df, x=value_column, nbins=bins, marginal="box",
                                  color_discrete_sequence=[PALETTE[1]])
            st.plotly_chart(style_chart(figure, "Distribution of " + value_column),
                            width="stretch")

            series = df[value_column].dropna()
            if len(series) > 0:
                skew = series.skew()
                shape = ("Right-skewed" if skew > 0.5
                         else "Left-skewed" if skew < -0.5 else "Roughly symmetric")
                row = st.columns(4)
                row[0].metric("Mean", f"{series.mean():,.2f}")
                row[1].metric("Median", f"{series.median():,.2f}")
                row[2].metric("Std Dev", f"{series.std():,.2f}")
                row[3].metric("Shape", shape)
        except Exception as error:
            st.error("This chart could not be drawn. Details: " + str(error))


# ------------------------ CHART 5 : SALES VS PROFIT --------------------------

def chart_5_sales_vs_profit(df, numerical, categorical, default_value, default_profit):
    with st.container(border=True):
        st.markdown("### 5️⃣  Sales vs Profit")
        st.caption("Scatter plot — the relationship between two numerical columns.")

        if not require(len(numerical) >= 2,
                       "A scatter plot needs at least two numerical columns. "
                       "This dataset has " + str(len(numerical)) + "."):
            return

        controls = st.columns(3)
        x_column = controls[0].selectbox(
            "X axis", numerical, index=index_of(numerical, default_value), key="c5_x")
        y_column = controls[1].selectbox(
            "Y axis", numerical, index=index_of(numerical, default_profit), key="c5_y")
        colour = controls[2].selectbox("Colour by", ["None"] + categorical, key="c5_colour")

        try:
            plot_data = df
            if len(df) > 5000:
                plot_data = df.sample(5000, random_state=42)
                st.caption("Showing a random sample of 5,000 points for speed.")

            figure = px.scatter(
                plot_data, x=x_column, y=y_column,
                color=None if colour == "None" else colour,
                opacity=0.6, color_discrete_sequence=PALETTE)
            st.plotly_chart(style_chart(figure, y_column + " vs " + x_column),
                            width="stretch")

            pair = df[[x_column, y_column]].dropna()
            if len(pair) > 1 and x_column != y_column:
                r = pair[x_column].corr(pair[y_column])
                if pd.notna(r):
                    strength = ("strong" if abs(r) >= 0.7
                                else "moderate" if abs(r) >= 0.4 else "weak")
                    direction = "positive" if r > 0 else "negative"
                    st.info("Correlation r = **" + str(round(r, 4)) + "** → a "
                            + strength + " " + direction + " relationship.")
        except Exception as error:
            st.error("This chart could not be drawn. Details: " + str(error))


# -------------------------- CHART 6 : CORRELATION ----------------------------

def chart_6_correlation(df, numerical):
    with st.container(border=True):
        st.markdown("### 6️⃣  Correlation Heatmap")
        st.caption("Shows how strongly every pair of numerical columns moves together.")

        if not require(len(numerical) >= 2,
                       "Correlation analysis requires at least two numerical columns. "
                       "This dataset has " + str(len(numerical)) + "."):
            return

        selected = st.multiselect("Columns to include", numerical,
                                  default=numerical[: min(8, len(numerical))],
                                  key="c6_columns")
        if len(selected) < 2:
            st.warning("Please select at least two columns.")
            return

        try:
            matrix = df[selected].corr().round(3)
            figure = px.imshow(matrix, text_auto=True, aspect="auto",
                               color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(style_chart(figure, "Correlation matrix"), width="stretch")
            st.caption(
                "r near +1 → both increase together.  r near −1 → one rises as the "
                "other falls.  r near 0 → no linear relationship."
            )
        except Exception as error:
            st.error("The heatmap could not be drawn. Details: " + str(error))


# =============================================================================
# SECTION 8 : PAGE 5  -  SIMILARITY ANALYSIS
# =============================================================================

def page_similarity():
    hero(
        "📐 Similarity Analysis",
        "Measure how similar or dissimilar any two records in the dataset are",
        ["Euclidean", "Manhattan", "Cosine"],
    )

    df = st.session_state.df_working
    if not require(df is not None, "Please load a dataset on the Home page first.", "info"):
        return

    types = detect_column_types(df)
    numerical = types["numerical"]
    if not require(len(numerical) >= 1,
                   "Similarity analysis needs at least one numerical column."):
        return

    st.markdown(
        "**Distance** measures *dissimilarity* — bigger means more different. "
        "**Cosine similarity** measures *similarity* — closer to 1 means more alike."
    )

    section("1 · Choose two records")
    last = len(df) - 1
    picker = st.columns(2)
    row_a = picker[0].number_input("Record 1 (row number)", 0, last, 0, 1)
    row_b = picker[1].number_input("Record 2 (row number)", 0, last, min(1, last), 1)
    if row_a == row_b:
        st.warning("Same record selected twice — distance will be 0 and cosine 1.")

    section("2 · Choose the attributes to compare")
    selected = st.multiselect("Numerical attributes", numerical,
                              default=numerical[: min(5, len(numerical))])
    if not require(len(selected) >= 1, "Please select at least one attribute."):
        return

    scale = st.checkbox(
        "Standardize before comparing (recommended)", value=True,
        help="Unit_Price reaches 95,000 while Discount only reaches 0.3. Without "
             "scaling, Unit_Price alone would decide the distance.",
    )

    subset = df[selected].copy()
    missing_here = int(subset.iloc[[row_a, row_b]].isna().sum().sum())
    if missing_here:
        st.info(str(missing_here) + " missing value(s) in these records were replaced "
                "with the column median before calculating.")
    subset = subset.fillna(subset.median(numeric_only=True)).dropna(axis=1, how="all")

    if subset.shape[1] == 0:
        st.warning("The selected attributes contain no usable values.")
        return

    try:
        if scale:
            prepared = pd.DataFrame(StandardScaler().fit_transform(subset),
                                    columns=subset.columns, index=subset.index)
        else:
            prepared = subset
    except Exception as error:
        st.error("The attributes could not be prepared. Details: " + str(error))
        return

    # reshape(1, -1) turns one row into the 2D array scikit-learn expects
    vector_a = prepared.iloc[row_a].to_numpy(dtype=float).reshape(1, -1)
    vector_b = prepared.iloc[row_b].to_numpy(dtype=float).reshape(1, -1)

    section("3 · The two records")
    table = pd.DataFrame({
        "Attribute": list(subset.columns),
        "Record " + str(row_a): df.iloc[row_a][list(subset.columns)].values,
        "Record " + str(row_b): df.iloc[row_b][list(subset.columns)].values,
    })
    if scale:
        table["Scaled " + str(row_a)] = vector_a[0].round(4)
        table["Scaled " + str(row_b)] = vector_b[0].round(4)
    st.dataframe(table, width="stretch", hide_index=True)

    section("4 · Results")
    try:
        euclidean = float(euclidean_distances(vector_a, vector_b)[0][0])
        manhattan = float(manhattan_distances(vector_a, vector_b)[0][0])
        # Cosine is undefined if either vector is all zeros
        if np.allclose(vector_a, 0) or np.allclose(vector_b, 0):
            cosine = np.nan
        else:
            cosine = float(cosine_similarity(vector_a, vector_b)[0][0])
    except Exception as error:
        st.error("The measures could not be calculated. Details: " + str(error))
        return

    results = st.columns(3)
    with results[0].container(border=True):
        st.metric("📏 Euclidean Distance", f"{euclidean:.4f}")
    with results[1].container(border=True):
        st.metric("🧱 Manhattan Distance", f"{manhattan:.4f}")
    with results[2].container(border=True):
        st.metric("🧭 Cosine Similarity",
                  "Not defined" if pd.isna(cosine) else f"{cosine:.4f}")

    if pd.isna(cosine):
        st.caption("Cosine similarity is undefined because one record became a zero "
                   "vector after scaling.")
    else:
        if cosine > 0.8:
            st.success("These two records are very similar.")
        elif cosine > 0.3:
            st.info("These two records are moderately similar.")
        elif cosine > -0.3:
            st.warning("These two records are almost unrelated.")
        else:
            st.warning("These two records point in opposite directions.")

    with st.expander("📖  Formulas used"):
        st.markdown(
            r"""
For two records **p** and **q** with *n* numerical attributes:

**Euclidean distance** (straight-line):
$$d(p,q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

**Manhattan distance** (city-block):
$$d(p,q) = \sum_{i=1}^{n}|p_i - q_i|$$

**Cosine similarity** (angle between the vectors):
$$\cos(p,q) = \frac{p \cdot q}{\|p\|\,\|q\|}$$
            """
        )

    with st.expander("🔢  Step-by-step calculation"):
        difference = vector_a[0] - vector_b[0]
        st.dataframe(
            pd.DataFrame({
                "Attribute": list(subset.columns),
                "p": vector_a[0].round(4),
                "q": vector_b[0].round(4),
                "p − q": difference.round(4),
                "|p − q|": np.abs(difference).round(4),
                "(p − q)²": (difference ** 2).round(4),
            }),
            width="stretch", hide_index=True,
        )
        st.markdown(
            "- Sum of (p − q)² = **" + str(round(float((difference ** 2).sum()), 4))
            + "** → Euclidean = square root = **" + str(round(euclidean, 4)) + "**\n"
            "- Sum of |p − q| = **" + str(round(float(np.abs(difference).sum()), 4))
            + "** → Manhattan = **" + str(round(manhattan, 4)) + "**"
        )


# =============================================================================
# SECTION 9 : SIDEBAR AND ROUTER
# =============================================================================

def build_sidebar():
    """Draw the sidebar and return the page the user selected."""
    st.sidebar.markdown("## 📊 Sales Data Mining")
    

    page = st.sidebar.radio(
        "Navigation",
        [" Home", " Dataset Overview", " Data Preprocessing",
         " Visualization", " Similarity Analysis"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    df = st.session_state.df_working
    if df is not None:
        st.sidebar.markdown("**Current dataset**")
        st.sidebar.caption(str(st.session_state.data_source))
        info = st.sidebar.columns(2)
        info[0].metric("Rows", f"{df.shape[0]:,}")
        info[1].metric("Cols", df.shape[1])
        st.sidebar.metric("Missing", f"{int(df.isna().sum().sum()):,}")

        st.sidebar.divider()
        st.sidebar.download_button(
            "⬇  Download Cleaned Dataset",
            data=csv_bytes(df),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            width="stretch",
        )
        if st.sidebar.button("↩  Reset to Original", width="stretch"):
            st.session_state.df_working = st.session_state.df_original.copy()
            st.session_state.action_log = []
            st.rerun()
    else:
        st.sidebar.info("No dataset loaded yet.")

    st.sidebar.divider()
    
    return page


def main():
    inject_custom_style()
    page = build_sidebar()

    if page.endswith("Home"):
        page_home()
    elif page.endswith("Dataset Overview"):
        page_overview()
    elif page.endswith("Data Preprocessing"):
        page_preprocessing()
    elif page.endswith("Visualization"):
        page_visualization()
    elif page.endswith("Similarity Analysis"):
        page_similarity()


# Streamlit runs this file top to bottom, so we call main() directly.
main()
