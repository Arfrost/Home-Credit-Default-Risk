import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data/raw/")
PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed/")
SQL_DIR = os.path.join(BASE_DIR, "sql/")
MODEL_DIR = os.path.join(BASE_DIR, "models/")

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5