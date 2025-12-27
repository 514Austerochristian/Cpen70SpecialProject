"""
Path configurations
"""

import os

# Get the absolute path to the directory where THIS file is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model paths
CNN_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'models', 'cnn_model_tuned.h5'))
LSTM_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'models', 'lstm_model_tuned.h5'))
HYBRID_MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'models', 'hybrid_model_tuned.h5'))

# Model comparison paths
MODEL_COMPARISON_PATHS = [
    '/workspaces/Cpen70SpecialProject/models/model_comparison.json',
    os.path.join(BASE_DIR, '..', 'models', 'model_comparison.json'),
    os.path.join(BASE_DIR, '..', 'data', 'processed', 'model_comparison.json'),
    'models/model_comparison.json',
    'data/processed/model_comparison.json'
]