# SynData AI

AI-Based Synthetic Dataset Generation and Evaluation Platform.

🌐 **Live Demo:** https://syn-data-ai.streamlit.app/

📦 **GitHub Repository:** https://github.com/Sahu-Yogesh/syn-data-ai

## Overview

**SynData AI** is an interactive Streamlit application developed for generating, analyzing, evaluating, and downloading synthetic tabular datasets.

The platform is designed to support experimentation with synthetic data generation techniques while providing tools for dataset exploration, model training, quality comparison, privacy screening, and result downloads.

## Key Features

- Interactive project dashboard
- Dataset analysis and exploratory data analysis
- Custom synthetic dataset generation
- CTGAN-based synthetic data generation
- Upload and process reference CSV datasets
- Select specific columns for model training
- Train, save, reuse, and generate synthetic records
- Statistical comparison between reference and synthetic data
- Correlation comparison
- Similarity and privacy screening
- Dynamic evaluation and visualization
- Synthetic dataset and evaluation result downloads

## Technology Stack

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **SDV**
- **CTGAN**
- **Scikit-learn**
- **Matplotlib**
- **Plotly**

## Project Structure

```text
syn-data-ai/
├── app.py
├── requirements.txt
├── .gitignore
├── models/
│   └── telco_metadata.json
└── src/
    ├── compare_data.py
    ├── correlation_comparison.py
    ├── ctgan_generator.py
    ├── custom_generator.py
    ├── eda.py
    ├── eda_summary.py
    ├── inspect_dataset.py
    ├── ml_utility_test.py
    ├── preprocess_dataset.py
    ├── privacy_similarity.py
    ├── sdv_quality_evaluation.py
    ├── statistical_comparison.py
    ├── test_ctgan.py
    ├── train_ctgan.py
    └── validate_synthetic.py
```

## Run the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/Sahu-Yogesh/syn-data-ai.git
cd syn-data-ai
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Streamlit

```bash
streamlit run app.py
```

The application will open in your browser, normally at:

```text
http://localhost:8501
```

## Suggested Workflow

1. Open the application dashboard.
2. Upload a reference CSV dataset through the appropriate module.
3. Review the dataset structure and columns.
4. Select the columns required for training.
5. Choose custom generation or CTGAN generation.
6. Train the model or reuse a saved model when available.
7. Generate synthetic records.
8. Evaluate the synthetic dataset using quality, statistical, correlation, similarity, and privacy tools.
9. Download the generated dataset and evaluation results.

## Deployment

The application is deployed using **Streamlit Community Cloud**.

Live application:

https://syn-data-ai.streamlit.app/

The deployment is connected to the GitHub repository. After making and testing code changes locally, push the changes to the `main` branch:

```bash
git add .
git commit -m "Describe your update"
git push
```

Streamlit Community Cloud can then detect the GitHub update and redeploy the application.

## Data and Privacy Notice

- Do not upload confidential, personally identifiable, regulated, or sensitive datasets to a public repository.
- Private datasets, generated outputs, and trained model artifacts should remain excluded through `.gitignore` where appropriate.
- Use anonymized or synthetic demonstration data when sharing screenshots, examples, or public demonstrations.
- Synthetic data generation does not automatically guarantee complete privacy. Generated datasets should be evaluated and reviewed before real-world use.

## Academic Project Context

This project was developed as an MSc IT project focused on synthetic tabular data generation, data quality evaluation, and privacy-aware experimentation.

## License

This project is intended for academic, learning, and research purposes. Add or update the license according to the project's distribution requirements.
