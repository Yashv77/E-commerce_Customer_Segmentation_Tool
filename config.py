import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model parameters
RANDOM_STATE = 42
N_INIT = 10

