from pathlib import Path
from datetime import datetime
import hashlib
import json
import re

import numpy as np
import pandas as pd
import streamlit as st

from src.custom_generator import generate_custom_dataset
from src.ctgan_generator import (
    train_ctgan_model,
    train_ctgan_with_metadata,
    load_ctgan_model,
    generate_ctgan_data,
    select_reference_columns,
)

try:
    from sdv.evaluation.single_table import evaluate_quality
except Exception:
    evaluate_quality = None


# ============================================================
# PATHS AND APPLICATION CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"
GENERATED_DIR = OUTPUT_DIR / "generated"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
EDA_DIR = OUTPUT_DIR / "eda"
MODEL_CATALOG_PATH = MODEL_DIR / "model_catalog.json"

for directory in [MODEL_DIR, GENERATED_DIR, EVALUATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="SynData AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
        padding: 14px;
        background: rgba(128,128,128,.04);
    }
    .hero {
        padding: 1.5rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(100,100,100,.12), rgba(100,100,100,.03));
        margin-bottom: 1rem;
    }
    .small-note { opacity: .75; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_filename(value: str, fallback: str = "dataset") -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_-")
    return value or fallback


def file_fingerprint(uploaded_file) -> str:
    content = uploaded_file.getvalue()
    return hashlib.sha256(content).hexdigest()[:16]


def dataframe_fingerprint(data: pd.DataFrame) -> str:
    try:
        raw = pd.util.hash_pandas_object(data, index=True).values.tobytes()
        return hashlib.sha256(raw).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(str(data.shape).encode()).hexdigest()[:16]


def read_csv_safely(path: Path):
    if not path.exists():
        return None, f"File not found: {path}"
    try:
        return pd.read_csv(path), None
    except Exception as error:
        return None, str(error)


def load_catalog() -> list:
    if not MODEL_CATALOG_PATH.exists():
        return []
    try:
        content = json.loads(MODEL_CATALOG_PATH.read_text(encoding="utf-8"))
        return content if isinstance(content, list) else []
    except Exception:
        return []


def save_catalog(catalog: list):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, default=str),
        encoding="utf-8",
    )


def register_model(model_path: Path, metadata: dict | None = None):
    catalog = load_catalog()
    model_path = model_path.resolve()
    entry = {
        "name": model_path.stem,
        "path": str(model_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if metadata:
        entry.update(metadata)

    catalog = [item for item in catalog if Path(item.get("path", "")).resolve() != model_path]
    catalog.append(entry)
    save_catalog(catalog)


def discover_models() -> list:
    models = []
    for path in sorted(MODEL_DIR.glob("*.pkl")):
        models.append(
            {
                "name": path.stem,
                "path": str(path.resolve()),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )

    catalog = {item.get("path"): item for item in load_catalog()}
    for item in models:
        saved = catalog.get(item["path"], {})
        item["dataset_name"] = saved.get("dataset_name", "Unknown")
        item["columns"] = saved.get("columns", [])
        item["rows"] = saved.get("rows", "Unknown")
        item["epochs"] = saved.get("epochs", "Unknown")
    return models


def clear_active_generation():
    for key in [
        "ctgan_synthesizer",
        "ctgan_model_path",
        "ctgan_generated_data",
        "current_real_data",
        "current_synthetic_data",
        "current_run_name",
        "current_run_type",
        "current_metadata",
    ]:
        st.session_state.pop(key, None)


def set_current_run(real_data, synthetic_data, run_name, run_type, model_path=None, metadata=None):
    st.session_state.current_real_data = real_data.copy()
    st.session_state.current_synthetic_data = synthetic_data.copy()
    st.session_state.current_run_name = run_name
    st.session_state.current_run_type = run_type
    if model_path:
        st.session_state.ctgan_model_path = str(model_path)
    if metadata is not None:
        st.session_state.current_metadata = metadata


def numeric_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=[np.number]).columns.tolist()


def categorical_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def comparable_columns(real: pd.DataFrame, synthetic: pd.DataFrame) -> list:
    return [column for column in real.columns if column in synthetic.columns]


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").dropna()


def distribution_similarity(real_values, synthetic_values) -> float | None:
    real_values = safe_numeric(real_values)
    synthetic_values = safe_numeric(synthetic_values)
    if real_values.empty or synthetic_values.empty:
        return None
    low = min(real_values.min(), synthetic_values.min())
    high = max(real_values.max(), synthetic_values.max())
    if low == high:
        return 1.0
    real_hist, _ = np.histogram(real_values, bins=20, range=(low, high), density=True)
    synth_hist, _ = np.histogram(synthetic_values, bins=20, range=(low, high), density=True)
    denominator = np.maximum(np.abs(real_hist), 1e-12)
    relative_error = np.mean(np.abs(real_hist - synth_hist) / denominator)
    return float(max(0.0, min(1.0, 1.0 - relative_error)))


def numerical_comparison(real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in comparable_columns(real, synthetic):
        r = safe_numeric(real[column])
        s = safe_numeric(synthetic[column])
        if r.empty or s.empty:
            continue
        similarity = distribution_similarity(r, s)
        rows.append({
            "Column": column,
            "Real_Mean": r.mean(),
            "Synthetic_Mean": s.mean(),
            "Mean_Difference": abs(r.mean() - s.mean()),
            "Real_Std": r.std(),
            "Synthetic_Std": s.std(),
            "Real_Min": r.min(),
            "Synthetic_Min": s.min(),
            "Real_Max": r.max(),
            "Synthetic_Max": s.max(),
            "Distribution_Similarity_Percent": round((similarity or 0) * 100, 2),
        })
    return pd.DataFrame(rows)


def categorical_comparison(real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in comparable_columns(real, synthetic):
        if column not in categorical_columns(real) and column not in categorical_columns(synthetic):
            continue
        real_freq = real[column].fillna("<MISSING>").astype(str).value_counts(normalize=True)
        synth_freq = synthetic[column].fillna("<MISSING>").astype(str).value_counts(normalize=True)
        categories = sorted(set(real_freq.index).union(synth_freq.index))
        tvd = 0.5 * sum(abs(real_freq.get(item, 0) - synth_freq.get(item, 0)) for item in categories)
        rows.append({
            "Column": column,
            "Unique_Real": int(real[column].nunique(dropna=False)),
            "Unique_Synthetic": int(synthetic[column].nunique(dropna=False)),
            "Distribution_Difference": round(float(tvd), 4),
            "Distribution_Similarity_Percent": round((1 - float(tvd)) * 100, 2),
        })
    return pd.DataFrame(rows)


def correlation_comparison(real: pd.DataFrame, synthetic: pd.DataFrame):
    columns = [c for c in comparable_columns(real, synthetic) if c in numeric_columns(real) and c in numeric_columns(synthetic)]
    if len(columns) < 2:
        return pd.DataFrame(), None
    real_corr = real[columns].corr(numeric_only=True)
    synth_corr = synthetic[columns].corr(numeric_only=True)
    common = real_corr.index.intersection(synth_corr.index)
    difference = (real_corr.loc[common, common] - synth_corr.loc[common, common]).abs()
    np.fill_diagonal(difference.values, 0)
    average_difference = float(difference.values.sum() / max(1, len(common) * (len(common) - 1)))
    return difference, average_difference


def exact_match_count(real: pd.DataFrame, synthetic: pd.DataFrame) -> int:
    columns = comparable_columns(real, synthetic)
    if not columns:
        return 0
    real_rows = set(map(tuple, real[columns].fillna("<MISSING>").astype(str).to_numpy()))
    synthetic_rows = synthetic[columns].fillna("<MISSING>").astype(str).to_numpy()
    return sum(tuple(row) in real_rows for row in synthetic_rows)


def evaluate_current_pair(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    result = {
        "real_rows": len(real),
        "synthetic_rows": len(synthetic),
        "real_columns": len(real.columns),
        "synthetic_columns": len(synthetic.columns),
        "common_columns": len(comparable_columns(real, synthetic)),
        "exact_matches": exact_match_count(real, synthetic),
    }
    num = numerical_comparison(real, synthetic)
    cat = categorical_comparison(real, synthetic)
    result["numeric_similarity"] = round(float(num["Distribution_Similarity_Percent"].mean()), 2) if not num.empty else None
    result["categorical_similarity"] = round(float(cat["Distribution_Similarity_Percent"].mean()), 2) if not cat.empty else None
    result["average_correlation_difference"] = correlation_comparison(real, synthetic)[1]
    return result


def run_sdv_quality(real: pd.DataFrame, synthetic: pd.DataFrame):
    if evaluate_quality is None:
        return None, "SDV quality evaluator is not available in this installation."
    try:
        from sdv.metadata import Metadata
        metadata = Metadata.detect_from_dataframe(
            data=real,
            table_name="evaluation_table",
        )
        report = evaluate_quality(
            real_data=real,
            synthetic_data=synthetic,
            metadata=metadata,
        )
        properties = report.get_properties()
        score = float(report.get_score())
        return {"overall": score, "properties": properties}, None
    except Exception as error:
        return None, str(error)


def save_generated_csv(data: pd.DataFrame, name: str) -> Path:
    path = GENERATED_DIR / f"{clean_filename(name)}.csv"
    data.to_csv(path, index=False)
    return path


def render_download(data: pd.DataFrame, filename: str, label="📥 Download CSV"):
    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def display_pair_summary(real: pd.DataFrame, synthetic: pd.DataFrame):
    metrics = evaluate_current_pair(real, synthetic)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Real Rows", f"{metrics['real_rows']:,}")
    c2.metric("Synthetic Rows", f"{metrics['synthetic_rows']:,}")
    c3.metric("Common Columns", metrics["common_columns"])
    c4.metric("Exact Matches", metrics["exact_matches"])


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🤖 SynData AI")
st.sidebar.caption("AI-Based Synthetic Dataset Generation")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "📊 Dataset Analysis",
        "🛠️ Custom Dataset Builder",
        "🤖 AI Generator",
        "📚 Model Library",
        "📈 Dynamic Evaluation",
        "🔐 Privacy Screening",
        "📥 Downloads",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### Technology")
st.sidebar.caption("Python • Pandas • NumPy")
st.sidebar.caption("SDV • CTGAN • Streamlit")
st.sidebar.caption("Dynamic evaluation and visualization")
st.sidebar.divider()
st.sidebar.info("Name: YOGESH SAHU - SMIT2627179\n\nProject :AI-Based Synthetic Dataset Generation")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🤖 SynData AI</h1>
        <p><b>AI-Based Synthetic Tabular Dataset Generation</b></p>
        <p class="small-note">Create custom datasets, reuse trained models, generate synthetic records, and evaluate results dynamically.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":
    st.header("📊 Project Dashboard")

    # The deployed app does not depend on a private/local reference CSV.
    # Reference data is supplied by the user through the relevant upload pages.
    real_data = None
    error = None
    models = discover_models()
    generated_files = list(GENERATED_DIR.glob("*.csv"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saved Models", len(models))
    c2.metric("Generated Datasets", len(generated_files))
    c3.metric("Reference Rows", f"{len(real_data):,}" if real_data is not None else "—")
    c4.metric("Reference Columns", len(real_data.columns) if real_data is not None else "—")

    st.subheader("🔄 Application Workflow")
    workflow = pd.DataFrame({
        "Step": ["1. Define / Upload", "2. Train or Reuse", "3. Generate", "4. Evaluate", "5. Download"],
        "Description": [
            "Create a schema or upload a reference CSV",
            "Reuse a saved model or train a new CTGAN model",
            "Generate synthetic records",
            "Compare distributions, correlations and similarity",
            "Export datasets and evaluation tables",
        ],
    })
    st.table(workflow)

    st.subheader("📌 Current Session")
    if "current_synthetic_data" in st.session_state:
        display_pair_summary(
            st.session_state.current_real_data,
            st.session_state.current_synthetic_data,
        )
        st.success(f"Active run: {st.session_state.get('current_run_name', 'Unnamed run')}")
        st.caption(f"Mode: {st.session_state.get('current_run_type', 'Unknown')}")
    else:
        st.info("No active generated dataset in this session. Use AI Generator or Custom Dataset Builder.")

    st.info(
        "No default reference dataset is loaded. Upload a CSV in Dataset Analysis "
        "or AI Generator when you want to work with reference data."
    )

    if models:
        st.subheader("📚 Available Models")
        st.dataframe(pd.DataFrame(models), use_container_width=True)


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "📊 Dataset Analysis":
    st.header("📊 Dataset Analysis")
    uploaded = st.file_uploader("Upload CSV for analysis", type=["csv"], key="analysis_upload")

    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            st.success("Dataset loaded successfully.")
        except Exception as error:
            st.error(f"Could not read the CSV: {error}")
            st.stop()
    else:
        st.info("Upload a CSV file to start analysis.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Columns", len(data.columns))
    c3.metric("Missing Values", int(data.isna().sum().sum()))
    c4.metric("Duplicate Rows", int(data.duplicated().sum()))

    st.subheader("Preview")
    st.dataframe(data.head(50), use_container_width=True)

    st.subheader("Column Information")
    info = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str).values,
        "Missing": [int(data[column].isna().sum()) for column in data.columns],
        "Unique Values": [int(data[column].nunique(dropna=False)) for column in data.columns],
    })
    st.dataframe(info, use_container_width=True)

    st.subheader("Numerical Statistics")
    numerical = data.select_dtypes(include=[np.number])
    if numerical.empty:
        st.info("No numerical columns found.")
    else:
        st.dataframe(numerical.describe().T, use_container_width=True)
        selected_num = st.selectbox("Choose a numerical column for a distribution chart", numerical.columns)
        hist = pd.DataFrame({"Value": numerical[selected_num].dropna()})
        if not hist.empty:
            bins = pd.cut(hist["Value"], bins=min(20, max(2, hist["Value"].nunique())), duplicates="drop").value_counts().sort_index()
            chart = pd.DataFrame({"Range": bins.index.astype(str), "Records": bins.values}).set_index("Range")
            st.bar_chart(chart)

    st.subheader("Categorical Distribution")
    categorical = categorical_columns(data)
    if categorical:
        selected_cat = st.selectbox("Choose a categorical column", categorical)
        counts = data[selected_cat].fillna("<MISSING>").astype(str).value_counts().head(20)
        st.bar_chart(counts)
        st.dataframe(counts.rename("Count"), use_container_width=True)
    else:
        st.info("No categorical columns found.")


# ============================================================
# CUSTOM DATASET BUILDER
# ============================================================

elif page == "🛠️ Custom Dataset Builder":
    st.header("🛠️ Custom Dataset Builder")
    st.write("Create a synthetic dataset without a reference CSV. This mode uses user-defined rules rather than CTGAN.")

    dataset_name = st.text_input("Dataset Name", "custom_dataset", key="custom_dataset_name")
    number_of_rows = st.number_input("Number of Rows", min_value=1, max_value=100000, value=1000, step=100, key="custom_rows")

    if "custom_columns" not in st.session_state:
        st.session_state.custom_columns = [{"name": "Age", "type": "Integer", "min": 18, "max": 65}]

    data_types = ["Integer", "Decimal", "Category", "Boolean", "Name", "Email", "Date", "Unique ID"]
    st.subheader("Define Columns")

    if st.button("➕ Add Column", key="add_custom_column"):
        index = len(st.session_state.custom_columns) + 1
        st.session_state.custom_columns.append({"name": f"Column_{index}", "type": "Integer", "min": 0, "max": 100})
        st.rerun()

    remove_index = None
    for index, column in enumerate(st.session_state.custom_columns):
        with st.expander(f"Column {index + 1}: {column.get('name', '')}", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                column["name"] = st.text_input("Column Name", column.get("name", f"Column_{index + 1}"), key=f"custom_name_{index}")
            with c2:
                selected_type = st.selectbox("Data Type", data_types, index=data_types.index(column.get("type", "Integer")), key=f"custom_type_{index}")
                column["type"] = selected_type

            if selected_type in ["Integer", "Decimal"]:
                c1, c2 = st.columns(2)
                with c1:
                    column["min"] = st.number_input("Minimum", value=float(column.get("min", 0)), key=f"custom_min_{index}")
                with c2:
                    column["max"] = st.number_input("Maximum", value=float(column.get("max", 100)), key=f"custom_max_{index}")
            elif selected_type == "Category":
                current = ", ".join(column.get("categories", ["A", "B", "C"]))
                text = st.text_input("Categories separated by commas", current, key=f"custom_categories_{index}")
                column["categories"] = [item.strip() for item in text.split(",") if item.strip()]
            elif selected_type == "Date":
                c1, c2 = st.columns(2)
                with c1:
                    column["start_date"] = st.date_input("Start Date", pd.Timestamp(column.get("start_date", "2020-01-01")).date(), key=f"custom_start_{index}")
                with c2:
                    column["end_date"] = st.date_input("End Date", pd.Timestamp(column.get("end_date", "2025-12-31")).date(), key=f"custom_end_{index}")

            if st.button("🗑️ Remove Column", key=f"remove_custom_{index}"):
                remove_index = index

    if remove_index is not None:
        st.session_state.custom_columns.pop(remove_index)
        st.rerun()

    if st.button("🚀 Generate Custom Dataset", type="primary", key="generate_custom"):
        try:
            generated = generate_custom_dataset(st.session_state.custom_columns, int(number_of_rows))
            st.session_state.custom_generated_data = generated
            st.session_state.custom_generated_data = generated
            st.session_state.current_synthetic_data = generated.copy()
            st.session_state.current_run_name = dataset_name
            st.session_state.current_run_type = "Custom Builder"
            path = save_generated_csv(generated, dataset_name)
            st.success(f"Generated {len(generated):,} rows and saved {path.name}.")
        except Exception as error:
            st.error(f"Generation failed: {error}")

    if "custom_generated_data" in st.session_state:
        generated = st.session_state.custom_generated_data
        st.subheader("Generated Dataset")
        st.dataframe(generated.head(50), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(generated):,}")
        c2.metric("Columns", len(generated.columns))
        render_download(generated, f"{clean_filename(dataset_name)}.csv")


# ============================================================
# AI GENERATOR
# ============================================================

elif page == "🤖 AI Generator":
    st.header("🤖 AI Generator — CTGAN")
    st.write("Reuse a saved model or train a new model from a reference CSV.")

    models = discover_models()
    tab_existing, tab_train = st.tabs(["📚 Use Existing Model", "🧠 Train New Model"])

    with tab_existing:
        if not models:
            st.info("No saved .pkl models were found in the models folder.")
        else:
            model_labels = [f"{item['name']} | {item['modified']} | {item['size_mb']} MB" for item in models]
            selected_label = st.selectbox("Choose a saved model", model_labels, key="existing_model_choice")
            selected_model = models[model_labels.index(selected_label)]
            st.write(f"**Model path:** `{selected_model['path']}`")
            st.write(f"**Training dataset:** {selected_model.get('dataset_name', 'Unknown')}")
            st.write(f"**Known columns:** {', '.join(selected_model.get('columns', [])) or 'Not recorded'}")
            rows = st.number_input("Synthetic rows", min_value=1, max_value=100000, value=1000, step=100, key="existing_rows")
            reference_upload = st.file_uploader("Optional reference CSV for evaluation", type=["csv"], key="existing_reference")

            if st.button("🚀 Load Model and Generate", type="primary", key="load_generate"):
                try:
                    synthesizer = load_ctgan_model(selected_model["path"])
                    synthetic = generate_ctgan_data(synthesizer, int(rows))
                    st.session_state.ctgan_synthesizer = synthesizer
                    st.session_state.ctgan_model_path = selected_model["path"]
                    real = None
                    if reference_upload is not None:
                        real = pd.read_csv(reference_upload)
                    if real is not None:
                        common = [c for c in real.columns if c in synthetic.columns]
                        if common:
                            real_for_eval = real[common].copy()
                            synthetic_for_eval = synthetic[common].copy()
                            set_current_run(real_for_eval, synthetic_for_eval, selected_model["name"], "Existing CTGAN Model", selected_model["path"])
                    st.session_state.ctgan_generated_data = synthetic
                    path = save_generated_csv(synthetic, f"{selected_model['name']}_generated")
                    st.success(f"Generated {len(synthetic):,} rows. Saved as {path.name}.")
                except Exception as error:
                    st.error(f"Could not load or generate: {error}")

    with tab_train:
        uploaded = st.file_uploader("Upload reference CSV", type=["csv"], key="train_reference")
        if uploaded is not None:
            try:
                reference_data = pd.read_csv(uploaded)
                st.success(f"Loaded {len(reference_data):,} rows and {len(reference_data.columns)} columns.")
                st.dataframe(reference_data.head(10), use_container_width=True)

                selected_columns = st.multiselect("Columns to use for training", list(reference_data.columns), default=list(reference_data.columns), key="train_columns")
                if selected_columns:
                    selected_reference_data = select_reference_columns(reference_data, selected_columns)
                    st.write(f"Selected columns: {len(selected_columns)}")

                    with st.expander("⚙️ Optional advanced metadata settings", expanded=False):
                        st.caption("Automatic detection is recommended for beginners. Change a type only when you understand the values in that column.")
                        sdtypes = ["numerical", "categorical", "datetime", "id", "boolean"]
                        metadata_settings = {}
                        for column in selected_columns:
                            series = selected_reference_data[column]
                            if pd.api.types.is_bool_dtype(series):
                                default = "boolean"
                            elif pd.api.types.is_numeric_dtype(series):
                                default = "numerical"
                            elif pd.api.types.is_datetime64_any_dtype(series):
                                default = "datetime"
                            else:
                                default = "categorical"
                            chosen = st.selectbox(f"Type for {column}", sdtypes, index=sdtypes.index(default), key=f"meta_type_{column}")
                            pii = st.checkbox(f"Contains personal information: {column}", value=False, key=f"meta_pii_{column}")
                            metadata_settings[column] = {"sdtype": chosen, "pii": pii}
                    
                    epochs = st.number_input("Training epochs", min_value=10, max_value=2000, value=300, step=50, key="train_epochs")
                    model_name = clean_filename(st.text_input("Model name", "custom_ctgan", key="train_model_name"))
                    model_path = MODEL_DIR / f"{model_name}.pkl"
                    rows = st.number_input("Synthetic rows after training", min_value=1, max_value=100000, value=1000, step=100, key="train_rows")

                    if model_path.exists():
                        st.warning("A model with this name already exists and will be replaced if you train.")

                    if st.button("🧠 Train, Save and Generate", type="primary", key="train_generate"):
                        try:
                            with st.spinner("Training CTGAN. This may take several minutes..."):
                                if "metadata_settings" in locals() and metadata_settings:
                                    synthesizer, metadata = train_ctgan_with_metadata(
                                        data=selected_reference_data,
                                        model_path=str(model_path),
                                        metadata_settings=metadata_settings,
                                        epochs=int(epochs),
                                    )
                                else:
                                    synthesizer, metadata = train_ctgan_model(
                                        data=selected_reference_data,
                                        model_path=str(model_path),
                                        epochs=int(epochs),
                                    )
                                synthetic = generate_ctgan_data(synthesizer, int(rows))

                            register_model(
                                model_path,
                                {
                                    "dataset_name": uploaded.name,
                                    "columns": selected_columns,
                                    "rows": len(selected_reference_data),
                                    "epochs": int(epochs),
                                },
                            )
                            st.session_state.ctgan_synthesizer = synthesizer
                            st.session_state.ctgan_model_path = str(model_path)
                            st.session_state.ctgan_generated_data = synthetic
                            set_current_run(selected_reference_data, synthetic, model_name, "New CTGAN Model", model_path, metadata)
                            output_path = save_generated_csv(synthetic, f"{model_name}_generated")
                            st.success(f"Model saved and {len(synthetic):,} rows generated. Output: {output_path.name}")
                        except Exception as error:
                            st.error(f"Training failed: {error}")
                else:
                    st.warning("Select at least one column.")
            except Exception as error:
                st.error(f"Could not read the CSV: {error}")
        else:
            st.info("Upload a reference CSV to train a new CTGAN model.")

    if "ctgan_generated_data" in st.session_state:
        st.divider()
        st.subheader("Latest Generated Data")
        generated = st.session_state.ctgan_generated_data
        st.dataframe(generated.head(50), use_container_width=True)
        render_download(generated, "synthetic_data.csv")


# ============================================================
# MODEL LIBRARY
# ============================================================

elif page == "📚 Model Library":
    st.header("📚 Model Library")
    st.write("All `.pkl` models found inside the models folder are listed here. You can reuse them without retraining.")
    models = discover_models()
    if not models:
        st.info("No saved models found in the models folder.")
    else:
        st.dataframe(pd.DataFrame(models), use_container_width=True)
        st.caption("Place an existing compatible SDV CTGAN `.pkl` file inside the models folder, then refresh the page.")


# ============================================================
# DYNAMIC EVALUATION
# ============================================================

elif page == "📈 Dynamic Evaluation":
    st.header("📈 Dynamic Evaluation")
    st.write("Evaluation is calculated from the active real and synthetic datasets. No fixed Telco scores are displayed.")

    real = st.session_state.get("current_real_data")
    synthetic = st.session_state.get("current_synthetic_data")

    if real is None or synthetic is None:
        st.info("Generate data first, or upload both datasets below.")
        c1, c2 = st.columns(2)
        with c1:
            real_upload = st.file_uploader("Real reference CSV", type=["csv"], key="eval_real_upload")
        with c2:
            synthetic_upload = st.file_uploader("Synthetic CSV", type=["csv"], key="eval_synth_upload")
        if real_upload is not None and synthetic_upload is not None:
            try:
                real = pd.read_csv(real_upload)
                synthetic = pd.read_csv(synthetic_upload)
                common = comparable_columns(real, synthetic)
                real = real[common].copy()
                synthetic = synthetic[common].copy()
                set_current_run(real, synthetic, "Uploaded Evaluation", "Manual Evaluation")
            except Exception as error:
                st.error(f"Could not load evaluation datasets: {error}")
                st.stop()
        else:
            st.stop()

    display_pair_summary(real, synthetic)
    st.caption(f"Active run: {st.session_state.get('current_run_name', 'Unnamed')}")

    st.subheader("🧪 SDV Quality Evaluation")
    sdv_result, sdv_error = run_sdv_quality(real, synthetic)
    if sdv_result is not None:
        st.metric("SDV Overall Quality", f"{sdv_result['overall'] * 100:.2f}%")
        st.dataframe(sdv_result["properties"], use_container_width=True)
    else:
        st.warning(sdv_error or "SDV quality evaluation could not be completed.")

    st.subheader("📊 Numerical Comparison")
    num = numerical_comparison(real, synthetic)
    if num.empty:
        st.info("No comparable numerical columns were found.")
    else:
        st.dataframe(num, use_container_width=True)
        st.bar_chart(num.set_index("Column")["Distribution_Similarity_Percent"])
        render_download(num, "numerical_comparison_dynamic.csv", "📥 Download Numerical Comparison")

    st.subheader("📊 Categorical Comparison")
    cat = categorical_comparison(real, synthetic)
    if cat.empty:
        st.info("No comparable categorical columns were found.")
    else:
        st.dataframe(cat, use_container_width=True)
        st.bar_chart(cat.set_index("Column")["Distribution_Similarity_Percent"])
        render_download(cat, "categorical_comparison_dynamic.csv", "📥 Download Categorical Comparison")

    st.subheader("🔗 Correlation Difference")
    corr_difference, average_difference = correlation_comparison(real, synthetic)
    if corr_difference.empty:
        st.info("At least two comparable numerical columns are required for correlation analysis.")
    else:
        st.metric("Average Absolute Correlation Difference", f"{average_difference:.4f}")
        st.dataframe(corr_difference, use_container_width=True)
        render_download(corr_difference.reset_index(), "correlation_difference_dynamic.csv", "📥 Download Correlation Difference")

    st.subheader("📈 Dynamic Distribution Graph")
    common_numeric = [c for c in comparable_columns(real, synthetic) if c in numeric_columns(real) and c in numeric_columns(synthetic)]
    if common_numeric:
        selected = st.selectbox("Choose a numerical column", common_numeric, key="eval_numeric_column")
        r = safe_numeric(real[selected])
        s = safe_numeric(synthetic[selected])
        low = min(r.min(), s.min())
        high = max(r.max(), s.max())
        if low != high:
            r_hist, edges = np.histogram(r, bins=20, range=(low, high))
            s_hist, _ = np.histogram(s, bins=edges)
            labels = [f"{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
            chart = pd.DataFrame({"Real": r_hist, "Synthetic": s_hist}, index=labels)
            st.bar_chart(chart)
    else:
        st.info("No numerical columns available for distribution comparison.")

    st.subheader("💾 Save Current Evaluation")
    if st.button("Save Evaluation Tables", key="save_dynamic_eval"):
        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        num.to_csv(EVALUATION_DIR / "numerical_comparison_dynamic.csv", index=False)
        cat.to_csv(EVALUATION_DIR / "categorical_comparison_dynamic.csv", index=False)
        if not corr_difference.empty:
            corr_difference.to_csv(EVALUATION_DIR / "correlation_difference_dynamic.csv")
        st.success(f"Evaluation tables saved in {EVALUATION_DIR}.")


# ============================================================
# PRIVACY SCREENING
# ============================================================

elif page == "🔐 Privacy Screening":
    st.header("🔐 Privacy Screening")
    st.write("This is a basic similarity screening tool, not a formal differential privacy guarantee.")

    real = st.session_state.get("current_real_data")
    synthetic = st.session_state.get("current_synthetic_data")

    if real is None or synthetic is None:
        st.info("Generate data first or upload real and synthetic CSV files.")
        c1, c2 = st.columns(2)
        with c1:
            real_upload = st.file_uploader("Real CSV", type=["csv"], key="privacy_real")
        with c2:
            synthetic_upload = st.file_uploader("Synthetic CSV", type=["csv"], key="privacy_synthetic")
        if real_upload is not None and synthetic_upload is not None:
            real = pd.read_csv(real_upload)
            synthetic = pd.read_csv(synthetic_upload)
            common = comparable_columns(real, synthetic)
            real = real[common].copy()
            synthetic = synthetic[common].copy()
            set_current_run(real, synthetic, "Privacy Upload", "Manual Privacy Screening")
        else:
            st.stop()

    matches = exact_match_count(real, synthetic)
    percentage = matches / max(1, len(synthetic)) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Real Rows", f"{len(real):,}")
    c2.metric("Synthetic Rows", f"{len(synthetic):,}")
    c3.metric("Exact Match Percentage", f"{percentage:.2f}%")

    if matches == 0:
        st.success("No exact row matches were found in the compared columns.")
    else:
        st.warning(f"{matches:,} synthetic rows exactly matched a real row in the compared columns.")

    st.subheader("Screening Limitations")
    st.warning(
        "Exact matching alone cannot prove that a dataset is private. "
        "Similarity may occur because the model learned common patterns. "
        "This screen is not a differential privacy guarantee."
    )


# ============================================================
# DOWNLOADS
# ============================================================

elif page == "📥 Downloads":
    st.header("📥 Downloads")
    st.write("Download generated datasets and saved evaluation files.")

    st.subheader("Generated Datasets")
    generated_files = sorted(GENERATED_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not generated_files:
        st.info("No generated CSV files found.")
    for path in generated_files:
        data, error = read_csv_safely(path)
        if data is not None:
            st.write(f"**{path.name}** — {len(data):,} rows × {len(data.columns)} columns")
            render_download(data, path.name, f"📥 Download {path.name}")

    st.subheader("Evaluation Files")
    evaluation_files = sorted(EVALUATION_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not evaluation_files:
        st.info("No evaluation CSV files found.")
    for path in evaluation_files:
        data, error = read_csv_safely(path)
        if data is not None:
            st.write(f"**{path.name}**")
            render_download(data, path.name, f"📥 Download {path.name}")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("SynData AI | MSc IT Research Project | Dynamic synthetic data generation and evaluation")
