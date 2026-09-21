import os

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer


def validate_reference_data(data):
    """Validate a reference DataFrame before model training."""
    if data is None or data.empty:
        raise ValueError("The reference dataset is empty.")
    if len(data.columns) < 1:
        raise ValueError("The dataset must contain at least one column.")
    if data.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    if len(data) < 10:
        raise ValueError("The reference dataset should contain at least 10 rows.")
    if any(str(column).strip() == "" for column in data.columns):
        raise ValueError("Column names cannot be empty.")
    return True


def select_reference_columns(data, selected_columns):
    """Return a copy containing only the selected columns."""
    validate_reference_data(data)
    if not selected_columns:
        raise ValueError("Please select at least one column.")
    missing = [column for column in selected_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    selected = data.loc[:, selected_columns].copy()
    if selected.shape[1] == 0:
        raise ValueError("No columns selected.")
    return selected


def create_metadata(data):
    """Detect and validate SDV metadata."""
    metadata = Metadata.detect_from_dataframe(
        data=data,
        table_name="synthetic_data_table",
    )
    metadata.validate()
    return metadata


def _build_synthesizer(metadata, epochs):
    return CTGANSynthesizer(
        metadata=metadata,
        epochs=int(epochs),
        verbose=True,
    )


def _save_synthesizer(synthesizer, model_path):
    model_path = os.fspath(model_path)
    directory = os.path.dirname(model_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    synthesizer.save(model_path)


def train_ctgan_model(data, model_path, epochs=300):
    """Train CTGAN with automatically detected metadata."""
    validate_reference_data(data)
    data = data.copy()
    metadata = create_metadata(data)
    synthesizer = _build_synthesizer(metadata, epochs)
    synthesizer.fit(data)
    _save_synthesizer(synthesizer, model_path)
    return synthesizer, metadata


def train_ctgan_with_metadata(data, model_path, metadata_settings, epochs=300):
    """Train CTGAN using simple user-reviewed SDV metadata settings."""
    validate_reference_data(data)
    data = data.copy()
    metadata = create_metadata(data)
    allowed_types = {"numerical", "categorical", "datetime", "id", "boolean"}

    for column_name, settings in (metadata_settings or {}).items():
        if column_name not in data.columns:
            raise ValueError(f"Column not found: {column_name}")
        sdtype = settings.get("sdtype")
        pii = bool(settings.get("pii", False))
        if sdtype not in allowed_types:
            raise ValueError(f"Unsupported semantic type for {column_name}: {sdtype}")
        metadata.update_column(
            column_name=column_name,
            sdtype=sdtype,
            pii=pii,
        )

    metadata.validate()
    synthesizer = _build_synthesizer(metadata, epochs)
    synthesizer.fit(data)
    _save_synthesizer(synthesizer, model_path)
    return synthesizer, metadata


def load_ctgan_model(model_path):
    """Load a previously saved SDV CTGAN model."""
    model_path = os.fspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return CTGANSynthesizer.load(model_path)


def generate_ctgan_data(synthesizer, num_rows):
    """Generate synthetic records from a trained synthesizer."""
    if int(num_rows) < 1:
        raise ValueError("Number of rows must be greater than zero.")
    return synthesizer.sample(num_rows=int(num_rows))
