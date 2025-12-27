# CPEN70 Special Project

This repository contains a Streamlit dashboard and supporting code for preparing data, training models, and visualizing model results for water quality prediction in Taal Lake.

## Quick Start

- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Run the dashboard (Streamlit):

```bash
streamlit run dashboard/app.py
```

## High-level Structure

Below is the repository layout with short descriptions for each top-level folder and key files.

```
README.md
requirements.txt
dashboard/
    app.py
    FOLDER READ.me
    config/
        __init__.py
        parameters.py
        settings.py
    data/
        __init__.py
        loader.py
        utils.py
    models/
        __init__.py
        comparison.py
        loader.py
        prediction.py
    pages/
        __init__.py
        home.py
        model_info.py
        visualization.py
    streamlit/
        config.toml
    ui/
        __init__.py
        components.py
        sidebar.py
    utils/
        __init__.py
        helpers.py
        paths.py
data/processed/
  locations_train.npy
  locations_test.npy
  model_processed_data.csv
  X_train.npy
  X_test.npy
  y_train.npy
  y_test.npy
data preparation/
  data preprocessing.py
  model data preparation.py
model training/
  cnn model training.py
  lstm model training.py
  hybrid model training.py
  model tuner.py
models/
  cnn_model.h5
  lstm_model.h5
  hybrid_model.h5
  model_comparison.json
processed data_ModelDataPreparation/
  processed_data.csv
raw data/
  climatic_parameters.csv
  water_parameters.csv

```

## Directories & Key Files

- **dashboard**: : Streamlit app entrypoint and dashboard UI.
  - **app.py**: Main Streamlit application.
  - **config/**: Dashboard configuration helpers (`parameters.py`, `settings.py`).

- **data**: Data loading and utility helpers.
  - **loader.py**: Functions to load processed and raw datasets.
  - **utils.py**: Data helper utilities.
  - **processed/**: Preprocessed NumPy arrays and CSV used for training and evaluation.

- **models**: Model training, loading and comparison logic.
  - **prediction.py**: Model inference helpers.
  - **comparison.py**: Scripts to compare model results and metrics.

- **pages**: Streamlit page modules (app pages) for `home`, `model_info`, and `visualization`.

- **ui**: Reusable UI components and sidebar definitions.

- **utils**: General helper functions and path utilities used across the codebase.

- **streamlit/config.toml**: Streamlit configuration for running the app.

## Model files & Tuning

- `models/` contains trained model artifacts (`*.h5`) and JSONs with tuning results. The `model tuning/` subtree stores tuning trials and Keras Tuner outputs.

## Data

- Raw input CSVs are located under `raw data/`.
- Processed datasets and arrays are available under `data/processed/` and `processed data_ModelDataPreparation/`.

## Development Notes

- Code is structured to separate data preparation, model training, and serving (dashboard).
- To add a new model, place training scripts in `model training/` and add loading/prediction helpers to `models/`.
